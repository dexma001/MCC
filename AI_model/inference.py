import os
import glob
import torch
import numpy as np
from models import PVTVAE
from dataset_pipeline import (PARENTS, BONE_NAMES, BONE_RADII, get_split_files,
                              make_run_name, find_run_dir_by_config, find_latest_checkpoint_in,
                              list_available_runs)
from physics_module import DifferentiablePhysics
# 추론 대상 실험도 train.py의 손실 가중치 + run 태그로 선택한다 (과거 실험 재현 가능).
from train import LAMBDA_RECON, LAMBDA_PHYS, BETA_KL, RUN_TAG

# 1. 환경 설정 및 디바이스 정의
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔥 추론 디바이스: {DEVICE}")

# 핵심 충돌 페어 정의 (검증용)
COLLIDING_PAIRS = [
    (('Hips', 'Chest'), ('LeftLowerArm', 'LeftHand')),   # 몸통 vs 왼팔
    (('Hips', 'Chest'), ('RightLowerArm', 'RightHand')), # 몸통 vs 오른팔
    (('LeftLowerArm', 'LeftHand'), ('RightLowerArm', 'RightHand')), # 왼팔 vs 오른팔
    (('LeftLowerLeg', 'LeftFoot'), ('RightLowerLeg', 'RightFoot'))  # 왼다리 vs 오른다리
]

def inference():
    # 2. 폴더 경로 탐색 (자동 감지)
    motions_dir = "../processed_motions_VMC" if os.path.exists("../processed_motions_VMC") else "processed_motions_VMC"

    # train.py에 설정된 손실 가중치/태그와 일치하는 run 폴더에서 마지막 epoch 가중치를 선택한다.
    run_dir = find_run_dir_by_config(LAMBDA_RECON, LAMBDA_PHYS, BETA_KL, tag=RUN_TAG)
    CHECKPOINT_PATH = find_latest_checkpoint_in(run_dir) if run_dir else None
    if not CHECKPOINT_PATH:
        target = make_run_name(LAMBDA_RECON, LAMBDA_PHYS, BETA_KL, tag=RUN_TAG)
        raise FileNotFoundError(
            f"설정과 일치하는 가중치 폴더(checkpoints/{target}/)를 찾을 수 없습니다.\n"
            f"사용 가능한 실험 폴더: {list_available_runs()}\n"
            f"(train.py의 LAMBDA_* 값을 위 이름 중 하나에 맞추거나 먼저 학습하세요.)")
    print(f"📦 가중치 로드: {os.path.relpath(CHECKPOINT_PATH)}")

    # 3. 학습한 Model (새 양식: 입력 87, 출력 84 쿼터니언)
    model = PVTVAE(input_dim=87, output_dim=84, latent_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval() # 평가(Inference) 모드로 전환
    print("성공: Trained model loaded.")

    physics_engine = DifferentiablePhysics(PARENTS, BONE_RADII).to(DEVICE)

    # ==========================================
    # 4. 테스트 데이터 로드 및 전처리
    # ==========================================
    # 🚨 과적합 방지: 학습에 쓰지 않은 held-out(test) 파일에서만 샘플을 고른다.
    pt_files = get_split_files(motions_dir, split='test')
    if not pt_files:
        raise FileNotFoundError(f" '{motions_dir}' 폴더에 held-out(test) .pt 파일이 없습니다.")

    # held-out 파일 중 무작위 1개로 단일 결과를 시각화한다.
    TEST_FILE_PATH = np.random.choice(pt_files)
    print(f"🎬 테스트 대상 모션 파일(held-out): {os.path.basename(TEST_FILE_PATH)}")
    
    original_motion = torch.load(TEST_FILE_PATH)  # [Frames, 87]

    # 모델 입력 스펙([Batch=1, Seq=30, 87])에 맞추기 위해 첫 30프레임 추출
    if original_motion.shape[0] < 30:
        raise ValueError(" 테스트 파일의 프레임 길이가 30보다 짧습니다.")

    input_sequence = original_motion[:30].unsqueeze(0).to(DEVICE)  # [1, 30, 87]

    # ==========================================
    # 5. AI 모델을 통한 모션 교정 (Inference)
    # ==========================================
    with torch.no_grad(): # 미분 계산을 꺼서 메모리를 절약하고 속도를 높입니다.
        recon_motion, _, _ = model(input_sequence)  # [1, 30, 87]

    # ==========================================
    # 6. 물리 엔진 검증 (Before & After 충돌 오차 비교) — FK 기반
    # ==========================================
    loss_phys_before = physics_engine.get_collision_loss_from_quats(
        input_sequence[..., :3], input_sequence[..., 3:], COLLIDING_PAIRS)
    loss_phys_after = physics_engine.get_collision_loss_from_quats(
        recon_motion[..., :3], recon_motion[..., 3:], COLLIDING_PAIRS)

    print("\n==============================================")
    print("물리 기반 모션 교정 평가 (Physics Evaluation)")
    print("==============================================")
    print(f"교정 전 원본 충돌 수치 (Before): {loss_phys_before.item():.6f}")
    print(f"교정 후 AI 결과 충돌 수치 (After) : {loss_phys_after.item():.6f}")
    print("==============================================")

    # 7. 시각화 툴에서 로드할 수 있도록 결과 저장
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)
    
    save_path_orig = os.path.join(output_dir, "sample_original.pt")
    save_path_corr = os.path.join(output_dir, "sample_corrected.pt")

    # 배치를 떼어내고 [30, 87] (Hips Pos 3 + Quats 84)로 CPU에 저장
    torch.save(input_sequence.squeeze(0).cpu(), save_path_orig)
    torch.save(recon_motion.squeeze(0).cpu(), save_path_corr)
    print(f" 결과 데이터가 '{output_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    inference()