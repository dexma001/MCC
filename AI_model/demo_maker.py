import os
import glob
import json
import datetime
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from models import PVTVAE
from dataset_pipeline import BONE_NAMES, BONE_MAP, PARENTS, BONE_RADII
from physics_module import DifferentiablePhysics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 30

# 주입 파라미터 (발표 데모용 — demo_results/demo_meta.json에 기록되어 동일 데모 재생성 가능)
INJECT_BONE = 'LeftUpperArm'
INJECT_AXIS = [0.0, 0.0, 1.0]     # 상박 로컬 Z축 기준 스윙(어덕션)
INJECT_MAX_DEG = 90.0
INJECT_FRAME_RANGE = (10, 25)     # [시작, 끝) 프레임 — sin 곡선으로 0 → 최대 → 0

# 주입 대상 뼈의 로컬 쿼터니언 슬라이스 (87차원 텐서 기준: 앞 3 = Hips Pos)
INJ_IDX = BONE_MAP[INJECT_BONE]
INJ_SLICE = slice(3 + INJ_IDX * 4, 3 + INJ_IDX * 4 + 4)


def create_extreme_demo():
    print("[발표용 데모 생성기] 극단적 클리핑 데이터 시뮬레이션을 시작합니다...")

    # 1. Trained Model (새 양식: 입력 87, 출력 84 쿼터니언)
    model = PVTVAE(input_dim=87, output_dim=84, latent_dim=64).to(DEVICE)
    ckpt_path = r"training_epoch_100_results/recon1_phys0.1_kl0.01/pvtvae_epoch_100.pth"
    if not os.path.exists(ckpt_path):
        print("가중치 파일을 찾을 수 없습니다.")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # 2. 대상 데이터 (새 양식 [Frames, 87]) — 주입 구간/모델 입력을 위해 30프레임 이상만 사용
    target_file = "" #"processed_motions_VMC/dataset-2_run_active_029.pt"
    if target_file and os.path.exists(target_file):
        full_motion = torch.load(target_file)
        if full_motion.shape[0] < SEQ_LEN:
            print(f"지정 파일의 프레임 수({full_motion.shape[0]})가 {SEQ_LEN} 미만입니다.")
            return
    else:
        cands = glob.glob("processed_motions_VMC/*.pt") or glob.glob("../processed_motions_VMC/*.pt")
        np.random.shuffle(cands)
        full_motion = None
        for cand in cands:
            t = torch.load(cand)
            if t.shape[0] >= SEQ_LEN:
                target_file, full_motion = cand, t
                break
        if full_motion is None:
            print(f"{SEQ_LEN}프레임 이상인 .pt 파일을 찾지 못했습니다.")
            return
    print(target_file)

    original_motion = full_motion[:SEQ_LEN].clone()  # [30, 87]

    # 3. 클리핑 버그 주입: INJECT_BONE 로컬 회전을 몸통 안쪽으로 강제로 꺾는다.
    #    (관절별 위치가 없는 새 양식에서는 쿼터니언만 수정 → FK가 하박/손까지 자동 전파)
    demo_motion = original_motion.clone()
    swing_axis = np.array(INJECT_AXIS)
    f_start, f_end = INJECT_FRAME_RANGE

    for f in range(f_start, f_end):
        ratio = np.sin((f - f_start) / float(f_end - f_start) * np.pi)   # 0 -> 1 -> 0 부드러운 곡선
        angle_deg = INJECT_MAX_DEG * ratio
        delta = R.from_rotvec(swing_axis * np.radians(angle_deg))

        q_local = demo_motion[f, INJ_SLICE].numpy()          # [qx,qy,qz,qw]
        q_new = (R.from_quat(q_local) * delta).as_quat()     # 로컬 프레임에서 추가 회전
        demo_motion[f, INJ_SLICE] = torch.tensor(q_new, dtype=demo_motion.dtype)

    # 3-1. 주입된 클리핑이 실제로 충돌을 만드는지 물리 엔진으로 확인
    #      (시각화 COMPARE 모드의 TEST_PAIRS와 동일한 4개 모니터링 페어 — 수치/화면 일치 보장)
    physics = DifferentiablePhysics(PARENTS, BONE_RADII)
    COLLIDING_PAIRS = [
        (('Hips', 'Chest'), ('LeftLowerArm', 'LeftHand')),
        (('Hips', 'Chest'), ('RightLowerArm', 'RightHand')),
        (('LeftLowerArm', 'LeftHand'), ('RightLowerArm', 'RightHand')),
        (('LeftLowerLeg', 'LeftFoot'), ('RightLowerLeg', 'RightFoot')),
    ]

    def depth_report(motion_87):
        """프레임별 최대 침투 깊이(cm) 시계열과 (최대 깊이, 충돌 프레임 수)를 반환."""
        dep = physics.get_penetration_depths_from_quats(
            motion_87[:, :3], motion_87[:, 3:], COLLIDING_PAIRS) * 100.0   # [F, P] cm
        fmax = dep.max(dim=1).values                                       # [F]
        return fmax, float(fmax.max()), int((fmax > 1e-4).sum())

    inj = physics.get_collision_loss_from_quats(
        demo_motion[:, :3], demo_motion[:, 3:], COLLIDING_PAIRS).item()
    _, maxpen_b, ncoll_b = depth_report(demo_motion)
    print(f"주입된 충돌(Before): 최대 침투 {maxpen_b:.2f} cm | 충돌 프레임 {ncoll_b}/{demo_motion.shape[0]} | loss={inj:.6f}")

    # 4. AI 교정 (Inference)
    input_tensor = demo_motion.unsqueeze(0).to(DEVICE)  # [1, 30, 87]
    with torch.no_grad():
        recon_motion, _, _ = model(input_tensor)
    corrected_motion = recon_motion.squeeze(0).cpu()    # [30, 87]

    corr_col = physics.get_collision_loss_from_quats(
        corrected_motion[:, :3], corrected_motion[:, 3:], COLLIDING_PAIRS).item()
    _, maxpen_a, ncoll_a = depth_report(corrected_motion)
    print(f"교정 후 충돌(After) : 최대 침투 {maxpen_a:.2f} cm | 충돌 프레임 {ncoll_a}/{corrected_motion.shape[0]} | loss={corr_col:.6f}")

    # 5. 시각화/평가 툴이 읽도록 동일 이름으로 저장 ([30, 87])
    os.makedirs("demo_results", exist_ok=True)
    torch.save(demo_motion, "demo_results/sample_original.pt")
    torch.save(corrected_motion, "demo_results/sample_corrected.pt")

    # 5-1. 데모 재현용 메타데이터 저장: 어떤 파일/가중치/주입 설정으로 만든 데모인지 기록
    #      (발표에서 잘 나온 데모를 나중에 target_file 고정으로 그대로 재생성할 수 있다)
    meta = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": str(target_file),
        "checkpoint": ckpt_path,
        "seq_len": SEQ_LEN,
        "inject": {
            "bone": INJECT_BONE,
            "local_axis": INJECT_AXIS,
            "max_deg": INJECT_MAX_DEG,
            "frame_range": list(INJECT_FRAME_RANGE),
            "profile": "sin(0 -> max -> 0)",
        },
        "collision_before": round(inj, 6),
        "collision_after": round(corr_col, 6),
        "max_pen_before_cm": round(maxpen_b, 2),
        "max_pen_after_cm": round(maxpen_a, 2),
        "collision_frames_before": ncoll_b,
        "collision_frames_after": ncoll_a,
    }
    with open("demo_results/demo_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("준비 완료! 이제 'python dataset_pipeline.py' (VISUALIZE=COMPARE)로 결과를 확인하세요.")


if __name__ == "__main__":
    create_extreme_demo()
