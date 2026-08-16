"""
모션 3D 시각화 전용 모듈 (구 dataset_pipeline.py의 MODE='VISUALIZE').

두 가지 뷰어를 제공한다. 둘 다 뼈대 위에 캡슐(살)을 겹쳐 그리고, 물리 엔진과 동일한
기준으로 침투 깊이를 색·수치로 표시한다 (판정·색 규칙은 한 곳에만 정의한다).
  - SINGLE : 처리된 .pt 모션 하나를 재생한다 (데이터 확인용).
             캡슐이 필요 없으면 --no-capsule (또는 SINGLE_SHOW_CAPSULE = False).
  - COMPARE: Before(손상 입력) / After(모델 출력)를 나란히 재생한다.

실행 방법은 두 가지이며, 둘 다 지원한다.

  (A) 소스 설정 + VS Code 실행 — 아래 '소스 설정' 블록의 VIEW_MODE 등을 편집하고
      VS Code에서 그냥 Run(Ctrl+F5)하거나 `python AI_model/viz_motion.py` 실행.
      인자를 하나도 주지 않으면 소스 설정이 그대로 적용된다.

  (B) 명령줄 인자 — 소스를 고치지 않고 일회성으로 모드를 바꾼다. 인자를 주면
      소스 설정보다 인자가 우선한다.
        python AI_model/viz_motion.py single           # 무작위 파일 재생
        python AI_model/viz_motion.py single <a.pt>    # 특정 파일 재생
        python AI_model/viz_motion.py single --no-capsule   # 뼈대 선만 (구판 동작)
        python AI_model/viz_motion.py compare                        # demo_results/
        python AI_model/viz_motion.py compare --results inference_results
        python AI_model/viz_motion.py compare --save demo_results/compare.gif

  .vscode/launch.json 에 모드별 디버그 구성(F5)도 등록되어 있다.

[분리 배경 — 2026-08-07]
  구판은 dataset_pipeline.py 하나에 MODE × VISUALIZE_TYPE 3중 분기가 중첩되어 있었고,
  모드를 바꾸려면 매번 소스를 편집해야 했다. 이제 모드는 CLI 인자로 고르며,
  학습 모듈(train.py 등)은 이 파일을 import하지 않으므로 matplotlib/scipy를
  더 이상 로드하지 않는다.

⚠️ 좌표 규약: 저장된 .pt는 Unity(Y-Up)다. FK 연산 전 반드시
   convert_unity_to_python_tensor / convert_offsets_to_python으로 Z-Up 변환을 거친다.
   이 변환은 '시각화 전용'이며 학습/평가 경로에는 절대 적용하지 않는다.
"""
import argparse
import glob
import os
import random
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

from dataset_pipeline import PARENTS, BONE_NAMES, BONE_MAP, BONE_RADII

# 경로 (프로젝트 루트 실행이 표준, AI_model/ 안에서 실행해도 되도록 폴백)
PROCESSED_PT_DIR = "processed_motions_VMC" \
    if os.path.exists("processed_motions_VMC") else "../processed_motions_VMC"
OFFSET_CSV_PATH = "Sample_Data/Standard_BoneOffsets.csv" \
    if os.path.exists("Sample_Data/Standard_BoneOffsets.csv") \
    else "../Sample_Data/Standard_BoneOffsets.csv"


# =====================================================================
# 소스 설정 — VS Code에서 인자 없이 Run(Ctrl+F5)할 때 쓰이는 값
# ---------------------------------------------------------------------
# 여기만 고치고 실행하면 된다. 명령줄 인자를 주면 아래 값 대신 인자가 쓰인다.
# (구판 dataset_pipeline.py의 VISUALIZE_TYPE / RESULTS_DIR / SAVE_ANIMATION_PATH 자리)
# =====================================================================
VIEW_MODE = "compare"       # "compare"(Before/After 비교) 또는 "single"(모션 하나 재생)

# --- SINGLE 모드 설정 ---
SINGLE_TARGET_FILE = ""     # 재생할 .pt 경로. 빈 문자열이면 무작위 추첨.
                            # 예: r"processed_motions_VMC/dataset-2_run_active_047.pt"
SINGLE_SHOW_CAPSULE = True  # 뼈대 위에 캡슐(살)을 겹쳐 그리고 자기충돌을 색·수치로 표시.
                            # False면 구판처럼 뼈대 선만 그린다 (--no-capsule과 동일).

# --- COMPARE 모드 설정 ---
COMPARE_RESULTS_DIR = "demo_results"   # demo_maker.py → "demo_results"
                                       # inference.py  → "inference_results"
SAVE_ANIMATION_PATH = ""    # 경로를 주면 gif로도 저장 (발표 슬라이드용).
                            # 예: "demo_results/compare.gif"

VIEW_MODES = ("single", "compare")


# ============================================================
# 뼈대 오프셋 / 좌표계 어댑터
# ============================================================
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


def convert_unity_to_python_tensor(motion_tensor):
    """
    시각화에서만 사용됩니다.
    [Frames, 87] Unity(Y-Up) 텐서를 FK 연산이 가능한 Python(Z-Up) 텐서로 완벽히 변환합니다.
    """
    converted = torch.zeros_like(motion_tensor)

    # 1. Hips Position: [x, y, z] -> [x, z, y]
    converted[:, 0] = motion_tensor[:, 0]   # px
    converted[:, 1] = motion_tensor[:, 2]   # pz (Unity z -> Python y)
    converted[:, 2] = motion_tensor[:, 1]   # py (Unity y -> Python z)

    # 2. Quaternions: [qx, qy, qz, qw] -> [-qx, -qz, -qy, qw]
    # 회전 방향(Handedness)과 축 매핑을 동시에 파이썬 기준으로 보정합니다.
    for i in range(21):
        s = 3 + i * 4
        converted[:, s+0] = -motion_tensor[:, s+0]   # -qx
        converted[:, s+1] = -motion_tensor[:, s+2]   # -qz
        converted[:, s+2] = -motion_tensor[:, s+1]   # -qy
        converted[:, s+3] = motion_tensor[:, s+3]    # qw

    return converted


def convert_offsets_to_python(offsets):
    """FK 연산을 위해 오프셋 [x, y, z]를 파이썬 기준 [x, z, y]로 변환합니다."""
    converted = {}
    for bone, vec in offsets.items():
        converted[bone] = torch.tensor([vec[0], vec[2], vec[1]], dtype=torch.float32)
    return converted


# ============================================================
# 순방향 운동학(FK) — 시각화용 뼈대 복원
# ============================================================
def compute_skeleton_positions(hips_pos, quats_84, offsets):
    quats = quats_84.view(len(BONE_NAMES), 4).numpy()
    global_pos = {}
    global_rot = {}

    global_pos['Hips'] = hips_pos.numpy()
    global_rot['Hips'] = R.from_quat(quats[BONE_MAP['Hips']])

    for child, parent in PARENTS.items():
        if parent is None:
            continue
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
# 관찰 전용 뷰어 컨트롤 (학습 코드와 의존성 없음)
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


def _load_offsets_or_exit():
    if not os.path.exists(OFFSET_CSV_PATH):
        print(f"❌ 오프셋 CSV를 찾을 수 없습니다: {OFFSET_CSV_PATH}")
        sys.exit(1)
    return load_standard_offsets(OFFSET_CSV_PATH)


# 그릴 뼈 세그먼트 (부모가 뼈대 안에 있는 것만) — 두 뷰어 공용
def _bones_to_draw():
    return [(PARENTS[b], b) for b in BONE_NAMES if PARENTS[b] in BONE_NAMES]


# ============================================================
# 캡슐(살) / 침투 판정 공용 정의 — SINGLE · COMPARE · viz_inject · viz_radii가 함께 쓴다
# ============================================================
# 각 캡슐이 실제로 차지하는 뼈 세그먼트만 하이라이트한다 (세그먼트는 자식 뼈 이름으로 식별).
# 예: Hips→Chest 캡슐 = Hips-Spine-Chest 구간 = 'Spine', 'Chest' 세그먼트.
# (기존에는 Neck/Head/Toes 등 캡슐 밖 세그먼트까지 빨갛게 표시되어 충돌 부위가 과장되었음)
CAPSULE_BONES = {
    'Hips_Chest': ['Spine', 'Chest'],
    'LeftArm': ['LeftHand'],        # LeftLowerArm→LeftHand 캡슐 = 팔꿈치-손 세그먼트
    'RightArm': ['RightHand'],
    'LeftLeg': ['LeftFoot'],        # LeftLowerLeg→LeftFoot 캡슐 = 무릎-발 세그먼트
    'RightLeg': ['RightFoot'],
}

# 시각화가 검사하는 4쌍 — train.py의 COLLIDING_PAIRS와 동일한 조합이어야 한다.
TEST_PAIRS = [
    ('Hips_Chest', 'LeftArm', 'Hips', 'Chest', 'LeftLowerArm', 'LeftHand'),
    ('Hips_Chest', 'RightArm', 'Hips', 'Chest', 'RightLowerArm', 'RightHand'),
    ('LeftArm', 'RightArm', 'LeftLowerArm', 'LeftHand', 'RightLowerArm', 'RightHand'),
    ('LeftLeg', 'RightLeg', 'LeftLowerLeg', 'LeftFoot', 'RightLowerLeg', 'RightFoot'),
]

DEPTH_FULL_RED = 0.02   # 침투 깊이 2cm 이상이면 최대 강도 빨강 (색 그라데이션 상한)

# 캡슐을 '굵은 반투명 선'으로 근사할 때의 굵기 배율 (BONE_RADII[자식] × LW_SCALE = 선 굵기 pt).
# ⚠️ 이건 화면 좌표(pt) 근사라 스크롤 줌을 해도 두께가 함께 커지지 않는다. 반지름을
#    '실제 기하'로 봐야 하면 viz_radii.py(캡슐 표면 메시)를 쓴다.
LW_SCALE = 400.0


def depth_style(depth, base_bone_color='dimgray', base_cap_color='dodgerblue'):
    """
    침투 깊이(m)에 비례한 (뼈 색, 캡슐 색, 캡슐 알파).
    얕은 접촉 = 옅은 빨강, 깊은 침투 = 진한 빨강.
    SINGLE/COMPARE 뷰어와 viz_inject.py · viz_radii.py가 공유한다 — 색 규칙이 갈라지면
    같은 깊이가 화면마다 다르게 보이므로 한 곳에서만 정의한다.
    """
    if depth <= 0.0:
        return base_bone_color, base_cap_color, 0.15
    t = min(depth / DEPTH_FULL_RED, 1.0)
    col = plt.cm.Reds(0.45 + 0.55 * t)
    return col, col, 0.25 + 0.45 * t


def make_penetration_probe():
    """
    관절 위치 텐서 [21, 3] → {뼈이름: 최대 침투 깊이(m)} 함수를 만들어 돌려준다.

    판정 기준은 물리 엔진과 동일하다(두 캡슐 끝점 뼈의 반지름 합). SINGLE/COMPARE가
    같은 함수를 쓰므로 한쪽만 조용히 다른 기준으로 색칠하는 일이 생기지 않는다.
    physics_module import를 이 안에 두어, 캡슐을 끄고 쓰는 경우 물리 모듈을 아예
    로드하지 않는다.
    """
    from physics_module import DifferentiablePhysics

    engine = DifferentiablePhysics(PARENTS, BONE_RADII).to('cpu')
    thresholds = [BONE_RADII[c1] + BONE_RADII[c2] for (_, _, _, c1, _, c2) in TEST_PAIRS]

    def probe(pos_t):
        depths = {}
        for (cap1, cap2, p1, c1, p2, c2), threshold in zip(TEST_PAIRS, thresholds):
            dist = engine.capsule_distance(
                pos_t[BONE_MAP[p1]], pos_t[BONE_MAP[c1]],
                pos_t[BONE_MAP[p2]], pos_t[BONE_MAP[c2]]
            ).item()
            pen = max(0.0, threshold - dist)
            if pen > 0:
                for b in CAPSULE_BONES[cap1] + CAPSULE_BONES[cap2]:
                    depths[b] = max(depths.get(b, 0.0), pen)
        return depths

    return probe


def set_panel_text(txt, depths):
    """패널 좌상단의 '최대 침투 깊이' 라벨을 갱신한다 (SINGLE/COMPARE 공용)."""
    max_pen = max(depths.values(), default=0.0)
    if max_pen > 0:
        txt.set_text(f"max penetration: {max_pen * 100:.1f} cm")
        txt.set_color('red')
    else:
        txt.set_text("no collision")
        txt.set_color('dimgray')


# ============================================================
# SINGLE 뷰어
# ============================================================
def visualize_single(target_file="", pt_dir=PROCESSED_PT_DIR,
                     show_capsule=SINGLE_SHOW_CAPSULE):
    """
    처리된 .pt 모션 하나를 3D로 재생한다. target_file이 없으면 무작위 선택.

    show_capsule=True면 뼈대 위에 캡슐(살)을 겹쳐 그리고, COMPARE와 같은 기준으로
    자기충돌 침투 깊이를 색·수치로 표시한다 — 원본 데이터가 이미 클리핑을 갖고 있는지
    한 화면에서 확인할 수 있다. 색·굵기·판정 기준은 모두 위 공용 정의를 쓴다.
    """
    offsets_py = convert_offsets_to_python(_load_offsets_or_exit())

    if target_file and os.path.exists(target_file):
        print(f"지정된 특정 파일 로드 중: {target_file}")
        motion = torch.load(target_file)
    else:
        if target_file:
            print(f"⚠️ 지정한 파일을 찾을 수 없습니다: {target_file} — 무작위로 대체합니다.")
        else:
            print("지정된 파일이 없어 무작위로 하나의 파일을 로드합니다.")
        # 전체 데이터셋을 로드하지 않고 파일 '경로'만 가져와서 무작위 선택
        pt_files = glob.glob(os.path.join(pt_dir, "*.pt"))
        if not pt_files:
            print(f"❌ 처리된 .pt 파일이 없습니다: {pt_dir} — 먼저 preprocess.py를 실행하세요.")
            sys.exit(1)
        random_file = random.choice(pt_files)
        print(f"무작위 선택된 파일: {os.path.basename(random_file)}")
        motion = torch.load(random_file)

    if motion.device.type != 'cpu':
        motion = motion.cpu()
    print(f"데이터 로드 완료: {motion.shape}")

    motion_py = convert_unity_to_python_tensor(motion)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Single Motion Viewer (Python Adapter Applied)"
                 + ("  —  capsule + self-collision" if show_capsule else ""), pad=10)

    bones_to_draw = _bones_to_draw()
    lines = [ax.plot([], [], [], 'o-', lw=2, color='blue')[0] for _ in bones_to_draw]

    # 캡슐(살) 레이어 — COMPARE와 같은 규약(굵기 = BONE_RADII[자식] × LW_SCALE).
    # 뼈대 선보다 먼저가 아니라 나중에 만들어도 알파가 낮아 뼈대를 가리지 않는다.
    capsules, probe, txt = [], None, None
    if show_capsule:
        probe = make_penetration_probe()
        for _parent_name, child_name in bones_to_draw:
            capsules.append(ax.plot([], [], [], '-', color='dodgerblue', alpha=0.15,
                                    lw=BONE_RADII.get(child_name, 0.05) * LW_SCALE)[0])
        txt = ax.text2D(0.02, 0.98, "", transform=ax.transAxes,
                        fontsize=11, va='top', fontweight='bold')

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(0, 1.6)
    ax.view_init(elev=15, azim=45)

    ax.set_xlabel('X (Left / Right)', fontsize=10, labelpad=10)
    ax.set_ylabel('Y (Depth / Unity Z)', fontsize=10, labelpad=10)
    ax.set_zlabel('Z (Height / Unity Y)', fontsize=10, labelpad=10)

    def update(frame_idx):
        # 어댑터를 거친 파이썬 전용 텐서를 사용하여 FK 연산을 안전하게 수행합니다.
        pos_t = get_pos_tensor(motion_py[frame_idx], offsets_py)
        frame_pos_np = pos_t.numpy()
        # 캡슐을 그릴 때만 침투를 재고, 그 결과로 뼈·캡슐 색을 정한다 (COMPARE와 같은 기준).
        depths = probe(pos_t) if probe is not None else {}
        for i, (parent_name, child_name) in enumerate(bones_to_draw):
            p_pos = frame_pos_np[BONE_MAP[parent_name]]
            c_pos = frame_pos_np[BONE_MAP[child_name]]
            xs, ys, zs = ([p_pos[0], c_pos[0]], [p_pos[1], c_pos[1]], [p_pos[2], c_pos[2]])
            lines[i].set_data(xs, ys)
            lines[i].set_3d_properties(zs)
            if not show_capsule:
                continue
            bone_col, cap_col, cap_alpha = depth_style(
                depths.get(child_name, 0.0), 'blue', 'dodgerblue')
            lines[i].set_color(bone_col)
            capsules[i].set_data(xs, ys)
            capsules[i].set_3d_properties(zs)
            capsules[i].set_color(cap_col)
            capsules[i].set_alpha(cap_alpha)
        if txt is not None:
            set_panel_text(txt, depths)
        return lines + capsules

    ani = FuncAnimation(fig, update, frames=motion_py.shape[0], interval=33, blit=False)
    # 관찰용 컨트롤: Pause/Resume 버튼(+스페이스바), 마우스 스크롤 줌
    # (반환값을 변수로 잡아둬야 콜백이 GC되지 않는다)
    _controls = attach_viewer_controls(fig, ani, [ax])
    plt.show()


# ============================================================
# COMPARE 뷰어 — Before/After + 침투 깊이 시각화
# ============================================================
def visualize_compare(results_dir="demo_results", save_path=""):
    """Before(손상)/After(교정) 모션을 나란히 재생하며 침투 깊이를 색·수치로 표시한다."""
    print("Before & After 비교 시각화 모드입니다.")

    offsets_py = convert_offsets_to_python(_load_offsets_or_exit())
    # 침투 판정 기준은 물리 엔진(get_collision_loss)과 동일하다 — 두 캡슐 끝점(자식 뼈)
    # 반지름의 합. SINGLE과 같은 probe를 쓴다 (기준이 갈라지지 않도록).
    probe = make_penetration_probe()

    if not os.path.isdir(results_dir) and os.path.isdir(os.path.join("..", results_dir)):
        results_dir = os.path.join("..", results_dir)
    orig_path = os.path.join(results_dir, "sample_original.pt")
    corr_path = os.path.join(results_dir, "sample_corrected.pt")

    if not os.path.exists(orig_path) or not os.path.exists(corr_path):
        print(f"'{results_dir}'에서 결과를 찾을 수 없습니다. "
              f"먼저 demo_maker.py(또는 inference.py)를 실행하세요.")
        sys.exit(1)

    motion_orig = torch.load(orig_path).cpu()
    motion_corr = torch.load(corr_path).cpu()

    # FK 연산 전 Python 좌표계 어댑터를 반드시 통과시킨다.
    motion_orig_py = convert_unity_to_python_tensor(motion_orig)
    motion_corr_py = convert_unity_to_python_tensor(motion_corr)

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

    bones_to_draw = _bones_to_draw()

    lines_orig_bone = [ax1.plot([], [], [], 'o-', lw=2.0, color='salmon')[0]
                       for _ in bones_to_draw]
    lines_corr_bone = [ax2.plot([], [], [], 'o-', lw=2.0, color='dodgerblue')[0]
                       for _ in bones_to_draw]

    lines_orig_capsule = []
    lines_corr_capsule = []

    for _parent_name, child_name in bones_to_draw:
        dynamic_lw = BONE_RADII.get(child_name, 0.05) * LW_SCALE
        lines_orig_capsule.append(
            ax1.plot([], [], [], '-', lw=dynamic_lw, color='salmon', alpha=0.15)[0])
        lines_corr_capsule.append(
            ax2.plot([], [], [], '-', lw=dynamic_lw, color='dodgerblue', alpha=0.15)[0])

    for ax in (ax1, ax2):
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(0, 1.8)
        ax.view_init(elev=15, azim=45)

    def update_compare(frame_idx):
        pos_orig_t = get_pos_tensor(motion_orig_py[frame_idx], offsets_py)
        pos_corr_t = get_pos_tensor(motion_corr_py[frame_idx], offsets_py)

        pos_orig_np = pos_orig_t.numpy()
        pos_corr_np = pos_corr_t.numpy()

        # 물리 엔진과 동일 기준의 침투 깊이 → 색 강도/패널 수치에 반영
        depths_orig = probe(pos_orig_t)
        depths_corr = probe(pos_corr_t)
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

        return (lines_orig_bone + lines_orig_capsule + lines_corr_bone
                + lines_corr_capsule + [txt_orig, txt_corr])

    ani = FuncAnimation(fig, update_compare, frames=motion_orig.shape[0],
                        interval=50, blit=False)
    if save_path:
        ani.save(save_path, writer='pillow', fps=20)
        print(f"애니메이션 저장 완료: {save_path}")
    # 관찰용 컨트롤은 '저장 후'에 부착한다 — 발표용 gif에 버튼 UI가 찍히지 않게.
    _controls = attach_viewer_controls(fig, ani, [ax1, ax2])
    plt.show()


def main(argv=None):
    """인자가 있으면 인자대로, 없으면 위 '소스 설정' 블록대로 뷰어를 띄운다."""
    # 소스 설정 오타를 조용히 넘기지 않는다 — 구판은 VISUALIZE_TYPE 오타가 아무 뷰어도
    # 띄우지 않고 조용히 종료했고, demo_maker의 DEMO_SCENARIO도 같은 함정을 갖고 있었다.
    if VIEW_MODE not in VIEW_MODES:
        raise ValueError(f"알 수 없는 VIEW_MODE: {VIEW_MODE!r} (지원: {list(VIEW_MODES)})")

    p = argparse.ArgumentParser(
        description="모션 3D 시각화 (single / compare). "
                    "인자를 생략하면 소스 상단의 '소스 설정' 블록 값을 사용한다.")
    sub = p.add_subparsers(dest="mode")

    p_single = sub.add_parser("single", help="처리된 .pt 모션 하나를 재생")
    p_single.add_argument("target", nargs="?", default=SINGLE_TARGET_FILE,
                          help="재생할 .pt 경로 (생략 시 소스 설정 → 무작위)")
    p_single.add_argument("--capsule", dest="capsule", action="store_true",
                          default=SINGLE_SHOW_CAPSULE,
                          help="캡슐(살) + 자기충돌 표시 (기본 켜짐)")
    p_single.add_argument("--no-capsule", dest="capsule", action="store_false",
                          help="뼈대 선만 그린다 (구판 동작)")

    p_cmp = sub.add_parser("compare", help="Before/After 비교 재생")
    p_cmp.add_argument("--results", default=COMPARE_RESULTS_DIR,
                       help="결과 폴더 (demo_maker → demo_results, inference → inference_results)")
    p_cmp.add_argument("--save", default=SAVE_ANIMATION_PATH,
                       help="지정 시 애니메이션을 gif로 저장 (예: demo_results/compare.gif)")

    args = p.parse_args(argv)
    # 서브커맨드를 안 줬으면 소스 설정의 VIEW_MODE를 따른다 (VS Code Run 경로).
    mode = args.mode or VIEW_MODE
    if not args.mode:
        print(f"ℹ️ 인자가 없어 소스 설정을 사용합니다 — VIEW_MODE='{mode}' "
              f"(모드를 바꾸려면 viz_motion.py 상단의 VIEW_MODE를 편집하세요).")

    if mode == "single":
        visualize_single(target_file=getattr(args, "target", SINGLE_TARGET_FILE),
                         show_capsule=getattr(args, "capsule", SINGLE_SHOW_CAPSULE))
    else:
        visualize_compare(results_dir=getattr(args, "results", COMPARE_RESULTS_DIR),
                          save_path=getattr(args, "save", SAVE_ANIMATION_PATH))


if __name__ == "__main__":
    main()
