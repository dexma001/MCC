import sys
import os
import glob
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PARENTS = {
    'Hips': None, 'Spine': 'Hips', 'Chest': 'Spine', 'Neck': 'Chest', 'Head': 'Neck',
    'LeftShoulder': 'Chest', 'LeftUpperArm': 'LeftShoulder', 'LeftLowerArm': 'LeftUpperArm', 'LeftHand': 'LeftLowerArm',
    'RightShoulder': 'Chest', 'RightUpperArm': 'RightShoulder', 'RightLowerArm': 'RightUpperArm', 'RightHand': 'RightLowerArm',
    'LeftUpperLeg': 'Hips', 'LeftLowerLeg': 'LeftUpperLeg', 'LeftFoot': 'LeftLowerLeg', 'LeftToes': 'LeftFoot',
    'RightUpperLeg': 'Hips', 'RightLowerLeg': 'RightUpperLeg', 'RightFoot': 'RightLowerLeg', 'RightToes': 'RightFoot'
}

BONE_NAMES = sorted(list(PARENTS.keys()))
BONE_MAP = {name: i for i, name in enumerate(BONE_NAMES)}
BONE_RADII = {
    'Hips': 0.06, 'Spine': 0.04, 'Chest': 0.08, 'Neck': 0.03, 'Head': 0.05,
    'LeftShoulder': 0.03, 'LeftUpperArm': 0.03, 'LeftLowerArm': 0.02, 'LeftHand': 0.02,
    'RightShoulder': 0.03, 'RightUpperArm': 0.03, 'RightLowerArm': 0.02, 'RightHand': 0.02,
    'LeftUpperLeg': 0.05, 'LeftLowerLeg': 0.04, 'LeftFoot': 0.03, 'LeftToes': 0.02,
    'RightUpperLeg': 0.05, 'RightLowerLeg': 0.04, 'RightFoot': 0.03, 'RightToes': 0.02
} # 뼈대 Capsulize

# [추가] 뼈대 고정 오프셋 데이터 로드 함수
def load_standard_offsets(offset_csv_path):
    """Standard_BoneOffsets.csv 파일을 읽어 관절별 고정 3D 오프셋 딕셔너리를 반환합니다."""
    df = pd.read_csv(offset_csv_path).set_index('BoneName')
    offsets = {}
    for bone in BONE_NAMES:
        if bone in df.index:
            row = df.loc[bone]
            # 오프셋 컬럼명(px, py, pz 또는 offset_x 등)에 맞춰 안전하게 파싱
            x = row.get('px', row.get('OffsetX', 0.0))
            y = row.get('py', row.get('OffsetY', 0.0))
            z = row.get('pz', row.get('OffsetZ', 0.0))
            offsets[bone] = torch.tensor([x, y, z], dtype=torch.float32)
        else:
            offsets[bone] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    return offsets

# PREPROCESS: CSV 데이터를 [Hips_Pos(3) + Quats(84) = 87] 텐서로 변환
def parse_csv_to_quaternion_tensor(motion_csv_path):
    """입력받은 모션 CSV 파일에서 Hips 위치와 전 관절 쿼터니언을 추출합니다."""
    df = pd.read_csv(motion_csv_path)
    frames = sorted(df['Frame'].unique())
    motion_list = []
    
    for f in frames:
        frame_data = df[df['Frame'] == f].set_index('BoneName')
        
        # 1. Hips의 동적 Position 추출 [3차원]
        hips_row = frame_data.loc['Hips']
        hips_pos = torch.tensor([hips_row['px'], hips_row['py'], hips_row['pz']], dtype=torch.float32) / 100.0
        
        # 2. 21개 전 관절의 Quaternion 추출 [21 * 4 = 84차원]
        quats_list = []
        for bone in BONE_NAMES:
            row = frame_data.loc[bone]
            q = torch.tensor([row['qx'], row['qy'], row['qz'], row['qw']], dtype=torch.float32)
            # 안전을 위한 쿼터니언 정규화
            q = q / (torch.norm(q) + 1e-8)
            quats_list.append(q)
            
        quats_tensor = torch.cat(quats_list) # [84]
        
        # 3. [3 + 84 = 87] 차원으로 결합
        frame_tensor = torch.cat([hips_pos, quats_tensor])
        motion_list.append(frame_tensor)
        
    return torch.stack(motion_list)

def convert_unity_to_python_tensor(motion_tensor):
    """
    VISUALIZE 모드에서만 사용됩니다.
    [Frames, 87] Unity(Y-Up) 텐서를 FK 연산이 가능한 Python(Z-Up) 텐서로 완벽히 변환합니다.
    """
    converted = torch.zeros_like(motion_tensor)
    
    # 1. Hips Position: [x, y, z] -> [x, z, y]
    converted[:, 0] = motion_tensor[:, 0] # px
    converted[:, 1] = motion_tensor[:, 2] # pz (Unity z -> Python y)
    converted[:, 2] = motion_tensor[:, 1] # py (Unity y -> Python z)
    
    # 2. Quaternions: [qx, qy, qz, qw] -> [-qx, -qz, -qy, qw]
    # 회전 방향(Handedness)과 축 매핑을 동시에 파이썬 기준으로 보정합니다.
    for i in range(21):
        s = 3 + i * 4
        converted[:, s+0]= -motion_tensor[:, s+0] # -qx
        converted[:, s+1] = -motion_tensor[:, s+2] # -qz
        converted[:, s+2] = -motion_tensor[:, s+1] # -qy
        converted[:, s+3] =  motion_tensor[:, s+3] # qw

    return converted

def convert_offsets_to_python(offsets):
    """FK 연산을 위해 오프셋 [x, y, z]를 파이썬 기준 [x, z, y]로 변환합니다."""
    converted = {}
    for bone, vec in offsets.items():
        converted[bone] = torch.tensor([vec[0], vec[2], vec[1]], dtype=torch.float32)
    return converted

# PYTHON 3D 시각화용: 오프셋 + 쿼터니언 순방향 운동학(FK) 뼈대 복원 함수
def compute_skeleton_positions(hips_pos, quats_84, offsets):
    quats = quats_84.view(len(BONE_NAMES), 4).numpy()
    global_pos = {}
    global_rot = {}
    
    global_pos['Hips'] = hips_pos.numpy()
    global_rot['Hips'] = R.from_quat(quats[BONE_MAP['Hips']])
    
    for child, parent in PARENTS.items():
        if parent is None: continue
        c_idx = BONE_MAP[child]
        local_rot = R.from_quat(quats[c_idx])
        child_offset = offsets[child].numpy()
        
        global_pos[child] = global_pos[parent] + global_rot[parent].apply(child_offset)
        global_rot[child] = global_rot[parent] * local_rot
        
    return global_pos

def get_pos_tensor(frame_tensor_87, offsets):
    hips_pos = frame_tensor_87[:3]
    quats_84 = frame_tensor_87[3:]
    pos_dict = compute_skeleton_positions(hips_pos, quats_84, offsets)
    
    pos_tensor = torch.zeros((len(BONE_NAMES), 3))
    for b, idx in BONE_MAP.items():
        pos_tensor[idx] = torch.tensor(pos_dict[b], dtype=torch.float32)
    return pos_tensor

# ============================================================
# 관찰 전용 뷰어 컨트롤 (VISUALIZE 모드 전용 — 학습 코드와 의존성 없음)
# ============================================================
def attach_viewer_controls(fig, ani, axes_3d):
    """
    matplotlib 애니메이션 뷰어에 관찰용 컨트롤을 붙인다 (SINGLE/COMPARE 공용).
      - Pause/Resume 버튼 (+ 스페이스바 단축키): 충돌이 일어나는 프레임을
        정지 상태로 자세히 관찰할 수 있다.
      - 마우스 스크롤: 3D 패널 확대/축소. 패널이 여러 개면(COMPARE) 두 패널을
        함께 줌하여 Before/After 비교 시점을 유지한다. 축소는 스크롤 다운.
    순수 matplotlib 이벤트만 사용한다 — 모델/학습 코드와 무관한 관찰 파트이므로
    어떤 학습 모듈도 import 하지 않는다 (위젯 import도 이 함수 안에서만).
    반환된 버튼 객체는 호출 측이 변수로 보관해야 콜백이 GC로 사라지지 않는다.
    """
    from matplotlib.widgets import Button

    state = {'paused': False}
    btn_ax = fig.add_axes([0.46, 0.02, 0.10, 0.05])   # 그림 하단 중앙
    btn = Button(btn_ax, 'Pause')

    def toggle_pause(_event=None):
        # FuncAnimation의 타이머(event_source)만 멈추고 살린다 — 정지 중에도
        # 스크롤 줌/시점 회전 등 다른 상호작용은 그대로 동작한다.
        if state['paused']:
            ani.event_source.start()
        else:
            ani.event_source.stop()
        state['paused'] = not state['paused']
        btn.label.set_text('Resume' if state['paused'] else 'Pause')
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == ' ':
            toggle_pause()

    def on_scroll(event):
        # 3D 패널 위에서 굴릴 때만 동작 (버튼/여백 위 스크롤은 무시)
        if event.inaxes not in axes_3d:
            return
        factor = 0.9 if event.button == 'up' else 1.1
        for ax in axes_3d:   # COMPARE: 두 패널 동기 줌 (같은 배율 유지)
            for get_lim, set_lim in ((ax.get_xlim3d, ax.set_xlim3d),
                                     (ax.get_ylim3d, ax.set_ylim3d),
                                     (ax.get_zlim3d, ax.set_zlim3d)):
                lo, hi = get_lim()
                center = (lo + hi) / 2.0
                half = (hi - lo) / 2.0 * factor
                set_lim(center - half, center + half)
        fig.canvas.draw_idle()

    btn.on_clicked(toggle_pause)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    return btn


# ============================================================
# Train / Test 분할 (과적합 검증용 held-out set)
# ============================================================
# 모든 스크립트(train / evaluate / inference)가 동일한 결정론적 분할을 공유하도록
# 고정 seed로 셔플한 뒤 앞쪽 VAL_RATIO 비율을 held-out(test)으로, 나머지를 train으로 사용.
VAL_RATIO = 0.1     # 전체의 10%를 학습에 쓰지 않고 평가 전용으로 격리
SPLIT_SEED = 42     # 분할 재현성을 위한 고정 시드

def get_split_files(pt_dir, split='train', val_ratio=VAL_RATIO, seed=SPLIT_SEED):
    """
    pt_dir 내 .pt 파일을 결정론적으로 train / test 로 나눈다.
      split='train' -> 학습용, 'test'(='val') -> 평가/추론용 held-out, 'all' -> 전체
    """
    files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))   # sorted로 순서 고정
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = int(len(files) * val_ratio)
    test_files = files[:n_val]
    train_files = files[n_val:]

    if split == 'train':
        return train_files
    if split in ('test', 'val'):
        return test_files
    return files


# ============================================================
# 체크포인트 폴더 관리 (실험용 가중치별로 폴더 분리)
# ============================================================
# 실험마다 checkpoints/ 아래에 사용된 손실 가중치를 이름으로 하는 하위 폴더를 만들어
# 가중치 파일 / run_config.json / log.txt 를 격리한다. 이렇게 하면 lambda 스윕 시
# 이전 실험 결과가 덮어써지지 않고, 폴더 이름만 봐도 어떤 설정인지 알 수 있다.

def make_run_name(lambda_recon, lambda_phys, beta_kl, tag=None):
    """
    실험 손실 가중치를 사람이 읽기 쉬운 폴더명으로 변환. 예: recon1_phys0.5_kl0.01
    tag를 주면 접두어로 붙는다 (예: tag='declip' → declip_recon1_phys0.5_kl0.01).
    → §4.1 디클리핑 시대의 run이 정규화 시대(무태그) 폴더와 절대 충돌하지 않게 하는 장치.
    """
    def fmt(v):
        return f"{float(v):g}"   # 1.0 -> '1', 0.5 -> '0.5', 0.01 -> '0.01'
    base = f"recon{fmt(lambda_recon)}_phys{fmt(lambda_phys)}_kl{fmt(beta_kl)}"
    return f"{tag}_{base}" if tag else base


def resolve_ckpt_root():
    """실행 위치(project root / AI_model)에 무관하게 checkpoints 루트 경로를 반환."""
    return "checkpoints" if os.path.exists("checkpoints") else "../checkpoints"


def find_latest_run_dir(ckpt_root=None):
    """
    가중치가 들어있는 run 하위 폴더 중 가장 최근에 학습된 폴더 경로를 반환한다.
    하위 폴더 구조가 없으면 구(舊) 평면 구조(ckpt_root 직속에 .pth) 호환으로 ckpt_root를 반환.
    아무 가중치도 없으면 None.
    """
    ckpt_root = ckpt_root or resolve_ckpt_root()
    run_dirs = [d for d in glob.glob(os.path.join(ckpt_root, "*"))
                if os.path.isdir(d) and glob.glob(os.path.join(d, "pvtvae_epoch_*.pth"))]
    if run_dirs:
        return max(run_dirs, key=os.path.getmtime)
    if glob.glob(os.path.join(ckpt_root, "pvtvae_epoch_*.pth")):
        return ckpt_root   # 구 평면 구조 호환
    return None


def find_latest_checkpoint_in(run_dir):
    """주어진 run 폴더에서 가장 마지막 epoch 가중치 경로를 반환 (없으면 None)."""
    ckpts = glob.glob(os.path.join(run_dir, "pvtvae_epoch_*.pth"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))


def find_run_dir_by_config(lambda_recon, lambda_phys, beta_kl, tag=None, ckpt_root=None):
    """
    손실 가중치 조합으로 특정 실험의 run 폴더를 직접 찾는다 (mtime이 아니라 설정으로 선택).
    train.py에서 쓰던 값과 동일한 (recon, phys, kl[, tag])을 주면 그 실험의 폴더를 반환하므로,
    가장 최근 실험이 아니라 '과거의 특정 실험'도 다시 평가할 수 있다.
    (tag=None이면 구(舊) 정규화-시대 폴더를, tag='declip'이면 디클리핑-시대 폴더를 찾는다.)
    가중치(.pth)가 들어있는 해당 폴더 경로를 반환하고, 없으면 None.
    """
    ckpt_root = ckpt_root or resolve_ckpt_root()
    run_dir = os.path.join(ckpt_root, make_run_name(lambda_recon, lambda_phys, beta_kl, tag=tag))
    if os.path.isdir(run_dir) and glob.glob(os.path.join(run_dir, "pvtvae_epoch_*.pth")):
        return run_dir
    return None


def list_available_runs(ckpt_root=None):
    """가중치가 들어있는 모든 run 폴더 이름 목록 (에러 메시지에서 사용자 안내용)."""
    ckpt_root = ckpt_root or resolve_ckpt_root()
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(ckpt_root, "*"))
                  if os.path.isdir(d) and glob.glob(os.path.join(d, "pvtvae_epoch_*.pth")))


# [수정] 머신러닝 데이터셋 클래스 (훈련 시에는 84차원 쿼터니언만 분리 제공)
class BandaiMotionDataset(Dataset):
    def __init__(self, pt_dir, seq_len=30, split='train'):
        # split='train'은 held-out test 파일을 제외한 학습용 파일만 로드
        self.pt_files = get_split_files(pt_dir, split=split)
        self.seq_len = seq_len
        self.data = []

        for f in self.pt_files:
            tensor = torch.load(f) # [Frames, 87]
            if tensor.shape[0] >= seq_len:
                self.data.append(tensor)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        tensor = self.data[idx]
        start = np.random.randint(0, tensor.shape[0] - self.seq_len + 1)
        window = tensor[start : start + self.seq_len] # [30, 87]

        # 새 양식: Hips Position(3) + 21 Quaternion(84) = 87 전체를 모델 입력으로 반환
        return window

# 3. main
if __name__ == "__main__":
    MODE = "VISUALIZE"

    # 경로 설정
    RAW_CSV_DIR = r"Sample_Data/Bandai_Dataset_csv_VMC_Normalized"
    PROCESSED_PT_DIR = "processed_motions_VMC"
    OFFSET_CSV_PATH = r"Sample_Data/Standard_BoneOffsets.csv"
    # COMPARE 모드가 읽을 결과 폴더 (demo_maker.py → "demo_results", inference.py → "inference_results")
    RESULTS_DIR = "demo_results"
    # 경로를 지정하면(예: "demo_results/compare.gif") COMPARE 애니메이션을 파일로도 저장 (발표 슬라이드용)
    SAVE_ANIMATION_PATH = ""

    # 고정 오프셋 뼈대 사전 로드
    if os.path.exists(OFFSET_CSV_PATH):
        STANDARD_OFFSETS = load_standard_offsets(OFFSET_CSV_PATH)
    else:
        print("Error / No_OFFSET")
        sys.exit()
        
    if MODE == "PREPROCESS":
        os.makedirs(PROCESSED_PT_DIR, exist_ok=True)
        csv_files = glob.glob(os.path.join(RAW_CSV_DIR, "**", "*.csv"), recursive = True)
        
        print(f"🔄 전처리 시작: 총 {len(csv_files)}개의 파일을 변환합니다.")
        for csv_path in csv_files:
            file_name = os.path.basename(csv_path).replace(".csv", ".pt")
            # 87차원 추출 함수 호출
            motion_tensor = parse_csv_to_quaternion_tensor(csv_path)
            torch.save(motion_tensor, os.path.join(PROCESSED_PT_DIR, file_name))
        print("✅ 모든 파일이 [Frames, 87] 차원의 .pt 텐서로 변환 완료되었습니다.")

    elif MODE == "VISUALIZE":
        """
        출력 전체 확인
        pt_files = glob.glob(os.path.join(PT_DIR, "*.pt"))
        motion = torch.load(pt_files[0]) # 전체 87프레임이 그대로 로드됨
        print(f"전체 데이터 로드 완료: {motion.shape}")
        """
        VISUALIZE_TYPE = "COMPARE" #SINGLE 또는 COMPARE
    
        if VISUALIZE_TYPE == "SINGLE":
            TARGET_FILE = '' #r"processed_motions_VMC/dataset-2_run_active_047.pt"

            if TARGET_FILE and os.path.exists(TARGET_FILE):
                print(f"지정된 특정 파일 로드 중: {TARGET_FILE}")
                motion = torch.load(TARGET_FILE)
            else:
                print("지정된 파일이 없어 무작위로 하나의 파일을 로드합니다.")
                #[수정된 부분] 전체 데이터셋을 로드하지 않고 파일 '경로'만 가져와서 무작위 선택
                pt_files = glob.glob(os.path.join(PROCESSED_PT_DIR, "*.pt"))
                
                if not pt_files:
                    print("에러: 처리된 .pt 파일이 없습니다.")
                    sys.exit()
                
                random_file = random.choice(pt_files)
                print(f"무작위 선택된 파일: {os.path.basename(random_file)}")
                motion = torch.load(random_file) 
            
            if motion.device.type != 'cpu': motion = motion.cpu()

            print(f"데이터 로드 완료: {motion.shape}")

            motion_py = convert_unity_to_python_tensor(motion)
            offsets_py = convert_offsets_to_python(STANDARD_OFFSETS)
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Single Motion Viewer (Python Adapter Applied)", pad=10)
            
            bones_to_draw = [(PARENTS[b], b) for b in BONE_NAMES if PARENTS[b] in BONE_NAMES]
            lines = [ax.plot([], [], [], 'o-', lw=2, color='blue')[0] for _ in bones_to_draw]

            ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(0, 1.6)
            ax.view_init(elev=15, azim=45)
            
            ax.set_xlabel('X (Left / Right)', fontsize=10, labelpad=10)
            ax.set_ylabel('Y (Depth / Unity Z)', fontsize=10, labelpad=10)
            ax.set_zlabel('Z (Height / Unity Y)', fontsize=10, labelpad=10)

            def update(frame_idx):
                # 어댑터를 거친 파이썬 전용 텐서를 사용하여 FK(순방향 운동학) 연산을 안전하게 수행합니다.
                frame_pos = get_pos_tensor(motion_py[frame_idx], offsets_py) 
                frame_pos_np = frame_pos.numpy()
                
                for i, (parent_name, child_name) in enumerate(bones_to_draw):
                    p_idx, c_idx = BONE_MAP[parent_name], BONE_MAP[child_name]
                    p_pos, c_pos = frame_pos_np[p_idx], frame_pos_np[c_idx]
                    
                    lines[i].set_data([p_pos[0], c_pos[0]], [p_pos[1], c_pos[1]])
                    lines[i].set_3d_properties([p_pos[2], c_pos[2]])
                return lines

            ani = FuncAnimation(fig, update, frames=motion_py.shape[0], interval=33, blit=False)
            # 관찰용 컨트롤: Pause/Resume 버튼(+스페이스바), 마우스 스크롤 줌
            viewer_controls = attach_viewer_controls(fig, ani, [ax])
            plt.show()
                
        elif VISUALIZE_TYPE == "COMPARE":
            print("Before & After 비교 시각화 모드입니다.")
            
            from physics_module import DifferentiablePhysics
            DEVICE = 'cpu'
            physics_engine = DifferentiablePhysics(PARENTS, BONE_RADII).to(DEVICE)

            orig_path = os.path.join(RESULTS_DIR, "sample_original.pt")
            corr_path = os.path.join(RESULTS_DIR, "sample_corrected.pt")

            if not os.path.exists(orig_path) or not os.path.exists(corr_path):
                print(f"'{RESULTS_DIR}'에서 결과를 찾을 수 없습니다. 먼저 demo_maker.py(또는 inference.py)를 실행하세요.")
                sys.exit()
                
            motion_orig = torch.load(orig_path).cpu()
            motion_corr = torch.load(corr_path).cpu()
            
            # 🚨 [핵심 수정] COMPARE 모드에서도 FK 연산 전 Python 좌표계 어댑터를 무조건 통과시킵니다.
            motion_orig_py = convert_unity_to_python_tensor(motion_orig)
            motion_corr_py = convert_unity_to_python_tensor(motion_corr)
            offsets_py = convert_offsets_to_python(STANDARD_OFFSETS)
            
            # 각 캡슐이 실제로 차지하는 뼈 세그먼트만 하이라이트한다 (세그먼트는 자식 뼈 이름으로 식별).
            # 예: Hips→Chest 캡슐 = Hips-Spine-Chest 구간 = 'Spine', 'Chest' 세그먼트.
            # (기존에는 Neck/Head/Toes 등 캡슐 밖 세그먼트까지 빨갛게 표시되어 충돌 부위가 과장되었음)
            CAPSULE_BONES = {
                'Hips_Chest': ['Spine', 'Chest'],
                'LeftArm': ['LeftHand'],        # LeftLowerArm→LeftHand 캡슐 = 팔꿈치-손 세그먼트
                'RightArm': ['RightHand'],
                'LeftLeg': ['LeftFoot'],        # LeftLowerLeg→LeftFoot 캡슐 = 무릎-발 세그먼트
                'RightLeg': ['RightFoot']
            }
            
            TEST_PAIRS = [
                ('Hips_Chest', 'LeftArm', 'Hips', 'Chest', 'LeftLowerArm', 'LeftHand'),
                ('Hips_Chest', 'RightArm', 'Hips', 'Chest', 'RightLowerArm', 'RightHand'),
                ('LeftArm', 'RightArm', 'LeftLowerArm', 'LeftHand', 'RightLowerArm', 'RightHand'),
                ('LeftLeg', 'RightLeg', 'LeftLowerLeg', 'LeftFoot', 'RightLowerLeg', 'RightFoot')
            ]

            # 침투 판정 기준을 물리 엔진(get_collision_loss)과 동일하게: 두 캡슐 끝점(자식 뼈) 반지름의 합.
            # (기존의 고정 0.1m 기준은 팔↔팔(0.04) / 다리↔다리(0.06) 페어를 과잉 표시했음)
            PAIR_THRESHOLDS = [BONE_RADII[c1] + BONE_RADII[c2] for (_, _, _, c1, _, c2) in TEST_PAIRS]
            DEPTH_FULL_RED = 0.02   # 침투 깊이 2cm 이상이면 최대 강도 빨강 (색 그라데이션 상한)

            fig = plt.figure(figsize=(14, 7))
            fig.suptitle("AI Motion Correction (Python Adapter Applied)", fontsize=16, fontweight='bold')

            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
            ax1.set_title("Before (Original / Injected)", fontsize=12)
            ax2.set_title("After (AI Corrected)", fontsize=12)

            # 프레임별 최대 침투 깊이(cm)를 패널 위에 수치로 표시 — 시각 결과와 수치가 항상 일치하도록
            txt_orig = ax1.text2D(0.02, 0.98, "", transform=ax1.transAxes,
                                  fontsize=11, va='top', fontweight='bold')
            txt_corr = ax2.text2D(0.02, 0.98, "", transform=ax2.transAxes,
                                  fontsize=11, va='top', fontweight='bold')

            bones_to_draw = [(PARENTS[b], b) for b in BONE_NAMES if PARENTS[b] in BONE_NAMES]

            lines_orig_bone = [ax1.plot([], [], [], 'o-', lw=2.0, color='salmon')[0] for _ in bones_to_draw]
            lines_corr_bone = [ax2.plot([], [], [], 'o-', lw=2.0, color='dodgerblue')[0] for _ in bones_to_draw]

            lines_orig_capsule = []
            lines_corr_capsule = []
            LW_SCALE = 400.0 
            
            for parent_name, child_name in bones_to_draw:
                actual_radius = BONE_RADII.get(child_name, 0.05) 
                dynamic_lw = actual_radius * LW_SCALE
                lines_orig_capsule.append(ax1.plot([], [], [], '-', lw=dynamic_lw, color='salmon', alpha=0.15)[0])
                lines_corr_capsule.append(ax2.plot([], [], [], '-', lw=dynamic_lw, color='dodgerblue', alpha=0.15)[0])

            for ax in [ax1, ax2]:
                ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(0, 1.8)
                ax.view_init(elev=15, azim=45)

            def pair_penetrations(pos_t):
                """
                현재 프레임의 관절 위치 텐서 [21, 3]에서 TEST_PAIRS별 침투 깊이(m)를 계산해
                {뼈이름: 최대 침투 깊이} 딕셔너리로 반환. 판정 기준은 물리 엔진과 동일(반지름 합).
                """
                depths = {}
                for (cap1, cap2, p1, c1, p2, c2), threshold in zip(TEST_PAIRS, PAIR_THRESHOLDS):
                    dist = physics_engine.capsule_distance(
                        pos_t[BONE_MAP[p1]], pos_t[BONE_MAP[c1]],
                        pos_t[BONE_MAP[p2]], pos_t[BONE_MAP[c2]]
                    ).item()
                    pen = max(0.0, threshold - dist)
                    if pen > 0:
                        for b in CAPSULE_BONES[cap1] + CAPSULE_BONES[cap2]:
                            depths[b] = max(depths.get(b, 0.0), pen)
                return depths

            def depth_style(depth, base_bone_color, base_cap_color):
                """침투 깊이(m)에 비례한 (뼈 색, 캡슐 색, 캡슐 알파). 얕은 접촉=옅은 빨강, 깊은 침투=진한 빨강."""
                if depth <= 0.0:
                    return base_bone_color, base_cap_color, 0.15
                t = min(depth / DEPTH_FULL_RED, 1.0)
                col = plt.cm.Reds(0.45 + 0.55 * t)
                return col, col, 0.25 + 0.45 * t

            def set_panel_text(txt, depths):
                max_pen = max(depths.values(), default=0.0)
                if max_pen > 0:
                    txt.set_text(f"max penetration: {max_pen * 100:.1f} cm")
                    txt.set_color('red')
                else:
                    txt.set_text("no collision")
                    txt.set_color('dimgray')

            def update_compare(frame_idx):
                pos_orig_t = get_pos_tensor(motion_orig_py[frame_idx], offsets_py)
                pos_corr_t = get_pos_tensor(motion_corr_py[frame_idx], offsets_py)

                pos_orig_np = pos_orig_t.numpy()
                pos_corr_np = pos_corr_t.numpy()

                # 물리 엔진과 동일 기준의 침투 깊이 → 색 강도/패널 수치에 반영
                depths_orig = pair_penetrations(pos_orig_t)
                depths_corr = pair_penetrations(pos_corr_t)
                set_panel_text(txt_orig, depths_orig)
                set_panel_text(txt_corr, depths_corr)

                for i, (parent_name, child_name) in enumerate(bones_to_draw):
                    p_idx, c_idx = BONE_MAP[parent_name], BONE_MAP[child_name]

                    # 시각화 배열로 위치 지정 (이미 Z-Up으로 맞춰져 있음)
                    po_p, po_c = pos_orig_np[p_idx], pos_orig_np[c_idx]
                    bone_col, cap_col, cap_alpha = depth_style(
                        depths_orig.get(child_name, 0.0), 'dimgray', 'salmon')
                    lines_orig_bone[i].set_data([po_p[0], po_c[0]], [po_p[1], po_c[1]])
                    lines_orig_bone[i].set_3d_properties([po_p[2], po_c[2]])
                    lines_orig_bone[i].set_color(bone_col)
                    lines_orig_capsule[i].set_data([po_p[0], po_c[0]], [po_p[1], po_c[1]])
                    lines_orig_capsule[i].set_3d_properties([po_p[2], po_c[2]])
                    lines_orig_capsule[i].set_color(cap_col)
                    lines_orig_capsule[i].set_alpha(cap_alpha)

                    pc_p, pc_c = pos_corr_np[p_idx], pos_corr_np[c_idx]
                    bone_col, cap_col, cap_alpha = depth_style(
                        depths_corr.get(child_name, 0.0), 'dodgerblue', 'dodgerblue')
                    lines_corr_bone[i].set_data([pc_p[0], pc_c[0]], [pc_p[1], pc_c[1]])
                    lines_corr_bone[i].set_3d_properties([pc_p[2], pc_c[2]])
                    lines_corr_bone[i].set_color(bone_col)
                    lines_corr_capsule[i].set_data([pc_p[0], pc_c[0]], [pc_p[1], pc_c[1]])
                    lines_corr_capsule[i].set_3d_properties([pc_p[2], pc_c[2]])
                    lines_corr_capsule[i].set_color(cap_col)
                    lines_corr_capsule[i].set_alpha(cap_alpha)

                return lines_orig_bone + lines_orig_capsule + lines_corr_bone + lines_corr_capsule + [txt_orig, txt_corr]

            ani = FuncAnimation(fig, update_compare, frames=motion_orig.shape[0], interval=50, blit=False)
            if SAVE_ANIMATION_PATH:
                ani.save(SAVE_ANIMATION_PATH, writer='pillow', fps=20)
                print(f"애니메이션 저장 완료: {SAVE_ANIMATION_PATH}")
            # 관찰용 컨트롤은 '저장 후'에 부착한다 — 발표용 gif에 버튼 UI가 찍히지 않게.
            viewer_controls = attach_viewer_controls(fig, ani, [ax1, ax2])
            plt.show()