import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Modules
from dataset_pipeline import (BandaiMotionDataset, PARENTS, BONE_RADII, BONE_NAMES, BONE_MAP,
                              make_run_name, find_latest_checkpoint_in)
from models import PVTVAE
from physics_module import DifferentiablePhysics

# 1. 하이퍼파라미터 및 환경 설정
# (이 모듈은 evaluate.py / inference.py 가 아래 LAMBDA_* 상수를 '가볍게' import 할 수 있어야 하므로,
#  데이터셋/모델 로딩 같은 무거운 초기화는 모두 train() 함수 안으로 옮겨 두었다.)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 학습 깊이 프로필: 100(빠른 람다 탐색) 또는 200(심화 학습).
# 100 epoch 스윕 결과 관찰 후 학습 깊이를 늘리기로 결정 → 200으로 전환 (2026-07-02).
# 이 값 하나만 바꾸면 아래 warmup/ramp/LR 스케줄이 모두 같은 비율로 자동 산출되므로,
# 100으로 되돌리면 이전(10/10/25) 설정이 정확히 재현된다.
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# =====================================================================
# [실험용 조절 손잡이] recon vs phys 가중치 (jitter loss는 제거됨)
#   - 실험마다 이 두 값을 바꿔가며 학습 -> inference.py -> evaluate.py 순서로 실행하면
#     evaluate.py가 지표를 evaluate_results.csv에 자동 기록합니다.
#   - 교정 강도는 절대값이 아니라 상대 비율(LAMBDA_PHYS / LAMBDA_RECON)이 결정하므로
#     LAMBDA_RECON은 1.0으로 고정(anchor)하고 LAMBDA_PHYS만 스윕하는 것이 표준 방식입니다.
#
#   [권장 스윕 그리드 — 로그 간격, EPOCHS 프로필별]
#     EPOCHS=100 : LAMBDA_PHYS ∈ {0.1, 0.3, 1.0, 3.0}          (상한 3.0)
#     EPOCHS=200 : LAMBDA_PHYS ∈ {0, 0.1, 0.3, 1.0, 3.0, 10.0} (상한 10.0)
#     - 0          : 물리 OFF 대조군(물리 손실의 효과를 인과적으로 입증하는 기준선)
#     - 0.3 ~ 1.0  : 충돌 제거와 동작 보존이 균형을 이루는 핵심 구간 (최적 후보)
#     - 상한값      : 학습 예산이 클수록 더 강한 제약을 소화할 수 있어 200ep에서는 10까지.
#                    수렴 실패/원본 희생이 보이면 그 지점이 상한 — 더 키우지 말 것.
#   (epoch=100 시절의 스윕 결과/가중치는 training_epoch_100_results/ 폴더에 보관됨)
# =====================================================================
LAMBDA_RECON = 1.0   # 원본 동작 보존(fidelity) 가중치 (anchor, 고정 권장)
LAMBDA_PHYS  = 10.0   # 충돌 제거(collision) 목표 가중치 — 위 스윕 그리드를 바꿔가며 실험
BETA_KL      = 0.01  # VAE 잠재공간 정규화 (아키텍처 손실, 실험 대상 아님)

# 물리 손실 커리큘럼: 처음엔 동작만 배우고, 이후 선형 램프로 LAMBDA_PHYS까지 상승.
# warmup/ramp/LR 주기는 EPOCHS에 비례(전체의 10% / 10% / 25%)해 자동 산출된다:
#   EPOCHS=100 -> 10 / 10 / 25  (기존 100ep 설정과 동일 재현)
#   EPOCHS=200 -> 20 / 20 / 50  (원래 200ep 설계와 동일)
# → "전체 학습의 ~80% 구간을 full 물리 가중치로 학습"하는 비율이 항상 유지된다.
PHYS_WARMUP_EPOCHS = EPOCHS // 10   # 이 에폭까지는 물리 손실 0 (순수 동작 학습)
PHYS_RAMP_EPOCHS   = EPOCHS // 10   # warmup 이후 0 -> LAMBDA_PHYS 로 선형 상승하는 구간 길이
LR_STEP_SIZE       = EPOCHS // 4    # StepLR 감쇠 주기 (전체의 25%마다 LR 절반)

# 전처리된 데이터 경로 (새 양식: [Frames, 87] = Hips Pos 3 + Quats 84)
PT_DIR = "processed_motions_VMC" if os.path.exists("processed_motions_VMC") else "../processed_motions_VMC"
OFFSET_CSV_PATH = "Sample_Data/Standard_BoneOffsets.csv"

# 2. 핵심 충돌 페어 정의 (Curriculum Learning)
# 전신을 모두 검사하면 느려지므로, 가장 잘 충돌하는 핵심 그룹만 묶어줍니다.
COLLIDING_PAIRS = [
    (('Hips', 'Chest'), ('LeftLowerArm', 'LeftHand')),   # 몸통 vs 왼팔
    (('Hips', 'Chest'), ('RightLowerArm', 'RightHand')), # 몸통 vs 오른팔
    (('LeftLowerArm', 'LeftHand'), ('RightLowerArm', 'RightHand')), # 왼팔 vs 오른팔
    (('LeftLowerLeg', 'LeftFoot'), ('RightLowerLeg', 'RightFoot'))  # 왼다리 vs 오른다리
]

# [참고] 새 양식에서는 뼈 길이가 고정 오프셋(Standard_BoneOffsets)으로 결정되어
# 구조적으로 항상 보존되므로 별도의 'bone length loss'(고무줄 팔 방지)는 불필요해졌다.

# 4. 메인 학습 루프
def train():
    print(f"🔥 학습 디바이스: {DEVICE}")

    # 3. 모델, 데이터, 물리 엔진 초기화 (무거운 로딩은 import 시가 아니라 학습 실행 시에만)
    # split='train': held-out 평가 파일을 제외한 학습용 파일만 사용 (과적합 검증 가능하도록)
    dataset = BandaiMotionDataset(PT_DIR, seq_len=30, split='train')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # 윈도우 에러 방지용 0
    print(f"📚 학습 파일 수(train split): {len(dataset)}")

    model = PVTVAE(input_dim=87, output_dim=84, latent_dim=64).to(DEVICE)
    physics_engine = DifferentiablePhysics(PARENTS, BONE_RADII, offset_csv_path=OFFSET_CSV_PATH).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # LR 감쇠 주기는 상단에서 EPOCHS에 비례해 산출된 LR_STEP_SIZE 사용 (100ep→25, 200ep→50)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.5)

    # 이번 실험의 손실 가중치를 이름으로 하는 전용 폴더에 저장한다.
    # 예: checkpoints/recon1_phys0.5_kl0.01/  → lambda 스윕 시 서로 덮어쓰지 않음.
    run_name = make_run_name(LAMBDA_RECON, LAMBDA_PHYS, BETA_KL)
    run_dir = os.path.join("checkpoints", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"📁 이번 실험 저장 폴더: {run_dir}")

    start_epoch = 1

    # 재개는 '같은 가중치' 폴더 안에서만 한다 (다른 lambda는 자기 폴더에서 새로 시작).
    latest_ckpt = find_latest_checkpoint_in(run_dir)
    if latest_ckpt:
        try:
            model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))
            start_epoch = int(os.path.basename(latest_ckpt).split('_')[2].split('.')[0]) + 1
            print(f"🔄 이전 학습 상태 로드 완료: {os.path.basename(latest_ckpt)} (Epoch {start_epoch}부터 재시작)")
        except RuntimeError as e:
            # 구(舊) 147차원 체크포인트는 새 87차원 모델과 호환되지 않음 → 새로 학습 시작
            print(f"⚠️ 기존 체크포인트({os.path.basename(latest_ckpt)})가 새 아키텍처와 호환되지 않아 무시하고 처음부터 학습합니다.")
            print(f"   (원인: {str(e).splitlines()[0]})")

    # 이번 학습에 사용된 실험용 가중치를 기록 (evaluate.py가 읽어 지표와 함께 기록)
    run_config = {
        "lambda_recon": LAMBDA_RECON,
        "lambda_phys": LAMBDA_PHYS,
        "beta_kl": BETA_KL,
        "phys_warmup_epochs": PHYS_WARMUP_EPOCHS,
        "phys_ramp_epochs": PHYS_RAMP_EPOCHS,
        "epochs": EPOCHS,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    log_file = os.path.join(run_dir, "log.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("🚀 Training Started\n")
        f.write(f"[config] LAMBDA_RECON={LAMBDA_RECON} LAMBDA_PHYS={LAMBDA_PHYS} BETA_KL={BETA_KL}\n")
        f.write("="*50 + "\n")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_recon_loss = 0
        total_kl_loss = 0
        total_phys_loss = 0

        # 단계적 물리 제약 (Curriculum Learning)
        # 처음 PHYS_WARMUP_EPOCHS 동안은 동작만 배우고(물리 0),
        # 이후 PHYS_RAMP_EPOCHS 구간에 걸쳐 0 -> LAMBDA_PHYS 로 선형 상승 후 유지.
        if epoch <= PHYS_WARMUP_EPOCHS:
            lambda_phys = 0.0
        else:
            ramp = min(1.0, (epoch - PHYS_WARMUP_EPOCHS) / max(1, PHYS_RAMP_EPOCHS))
            lambda_phys = LAMBDA_PHYS * ramp

        for batch_data in dataloader:
            # batch_data: [Batch, 30, 87] (Hips Pos 3 + Quats 84)
            batch_data = batch_data.to(DEVICE)

            optimizer.zero_grad()

            # 1. Forward — 출력도 [Batch, 30, 87] (Hips 통과 + 교정된 Quats)
            recon_motion, mu, logvar = model(batch_data)

            # 위치/회전 분리
            hips_pos     = batch_data[..., :3]     # [B, 30, 3]  루트 위치(원본)
            orig_quats   = batch_data[..., 3:]     # [B, 30, 84]
            recon_quats  = recon_motion[..., 3:]   # [B, 30, 84]

            # 🚨 1. 회전(쿼터니언) 복원 Loss — 원본 자세를 최대한 보존
            loss_recon_quat = nn.MSELoss()(recon_quats, orig_quats)

            # 🚨 2. L1 희소성(Sparsity) Loss — 교정량(delta)이 0에 가깝도록 유도
            #    (뼈 길이는 고정 오프셋 FK로 구조적으로 보존되므로 bone-length loss는 제거됨)
            loss_sparsity = nn.L1Loss()(recon_quats, orig_quats) * 0.05

            # Total Recon
            loss_recon = loss_recon_quat + loss_sparsity
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # 물리 엔진 연동: FK로 관절 월드 좌표 복원 → 충돌 Loss
            loss_phys = torch.tensor(0.0, device=DEVICE)
            if lambda_phys > 0:
                global_pos = physics_engine.forward_kinematics(hips_pos, recon_quats)
                loss_phys = physics_engine.get_collision_loss(global_pos, COLLIDING_PAIRS)

            # 최종 역전파 (jitter loss 제거됨: recon vs phys 두 축만 사용)
            loss = (LAMBDA_RECON * loss_recon) + (BETA_KL * loss_kl) + (lambda_phys * loss_phys)
            loss.backward()
            
            # 💥 안전벨트 추가: 기울기(Gradient)가 폭발하지 않도록 최대치 제한
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_recon_loss += loss_recon.item()
            total_kl_loss += loss_kl.item()
            total_phys_loss += loss_phys.item()


        # 에폭 결과 출력
        num_batches = len(dataloader)
        log_msg = (f"Epoch [{epoch}/{EPOCHS}] "
                   f"Recon: {total_recon_loss/num_batches:.4f} (λ={LAMBDA_RECON:.3f}) | "
                   f"KL: {total_kl_loss/num_batches:.4f} | "
                   f"Phys: {total_phys_loss/num_batches:.4f} (λ={lambda_phys:.3f})")
        
        # 터미널 화면에 출력
        print(log_msg)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + f" | LR: {current_lr:.6f}\n")
            
        # 모델 체크포인트 저장 (10 에폭마다)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(run_dir, f"pvtvae_epoch_{epoch}.pth"))
            ckpt_msg = f"💾 Checkpoint saved: {run_name}/pvtvae_epoch_{epoch}.pth"
            print(ckpt_msg)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(ckpt_msg + "\n")

if __name__ == "__main__":
    train()