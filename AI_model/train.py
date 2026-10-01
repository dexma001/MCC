import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_pipeline import (BandaiMotionDataset, PARENTS, BONE_RADII, RADII_MODE, RADII_TAG,
                              SOFT_TISSUE_KAPPA, SKELETON_STATURE_M,
                              make_run_name, resolve_ckpt_root, find_latest_checkpoint_in)
from physics_module import DifferentiablePhysics
import corruption
from models import TransformerDenoiser

# 1. 하이퍼파라미터 및 환경 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS = 2500        # [2026-09-22] 목표 에폭. l10.1_phys0.5 를 1500 → 2500 으로 이어 학습하며 관찰한다.
                     #   재개는 폴더의 마지막 체크포인트에서 이어가므로 이 값은 '어디까지 갈지'만 정한다.
                     #   더 늘릴 때는 이 값만 올린다 — LR/커리큘럼 스케줄은 아래에서 EPOCHS 와 분리돼 있다.
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# [2026-09-28] 손상 주입을 어디서 돌릴지. True(기본) = DEVICE(GPU)에서 배치마다 직접 호출.
#   실측(batch 32, 87배치/에폭): CPU+프리페치 13.2~13.8 s/에폭·CPU 코어 6.2개 상시 점유
#   → GPU 직접 6.5~6.7 s/에폭·코어 1.1개. 1에폭 전체에서 난수 소비·주입 유형·폴백이 CPU 경로와
#   완전히 같고 출력 차이는 최대 1.8e-7(부동소수점 잡음)이다.
#   False = 구 경로(CPU 주입 + 백그라운드 프리페치). 되돌릴 때는 이 값만 바꾼다.
INJECT_ON_DEVICE = True

# =====================================================================
# [실험용 조절 손잡이] recon vs phys 가중치
#   - 실험마다 이 두 값을 바꿔가며 train.py -> evaluate.py 순서로 실행하면
#     evaluate.py가 지표를 evaluate_results.csv에 자동 기록합니다.
#     (inference.py는 CSV에 기록하지 않는 단발 시각화용이라 이 경로에 필요하지 않다.)
#   - 교정 강도는 절대값이 아니라 상대 비율(LAMBDA_PHYS / LAMBDA_RECON)이 결정하므로
#     LAMBDA_RECON은 1.0으로 고정(anchor)하고 LAMBDA_PHYS만 스윕하는 것이 표준 방식입니다.
# 
#   BETA_KL 주의: 이 아키텍처(TransformerDenoiser)에는 KL 손실이 '존재하지 않는다'.
#   아래 상수는 오직 run 폴더 이름(make_run_name)과 공유 평가 코드의 import 호환을
#   위해서만 존재하며, 손실 계산에는 절대 쓰이지 않는다. 0.0 고정 — 바꾸지 말 것.
#   (KL을 실제로 쓰는 PVTVAE 실험을 재현하려면 PVTVAE_baseline/train.py 를 사용한다.)
# =====================================================================
LAMBDA_RECON = 1.0   # 복구 타겟(클린 정답) 충실도 가중치 (anchor, 고정 권장)
LAMBDA_PHYS  = 0.5   # 충돌 제거(collision) 목표 가중치 — 동결 v1 값 (2026-09-11 복원)
BETA_KL      = 0.0   # [주의] 손실 아님 — run 폴더 이름/평가 코드 호환용 상수 (0.0 고정)

# =====================================================================
# [§4.1 지도학습 디클리핑] — 근본 원인 보고서(collision_after_root_cause_report.md)의 처방.
#   기존 클린→클린 학습은 항등 함수로 수렴하여 입력의 충돌을 제거하지 못했다 (실증됨).
#   이제 매 배치에서 '입력'에만 클리핑을 주입하고(corruption.py), 손실은 '클린 원본'과
#   계산한다 → loss_recon이 '손상 → 클린' 복구 사상을 직접 지도하고, loss_phys가
#   마침내 0이 아닌 값(≈0.03 영역)을 가져 LAMBDA_PHYS가 실제 그래디언트를 스케일한다.
#
#   혼합비 (2026-07-07/08 합의, 보고서 §7-B): 클린 50% / 손상 50%,
#   손상 중 일시적 30% : 지속적 70% (전체 분포 = 클린 50 / 일시적 15 / 지속적 35).
#   지속 주입 = 전-윈도우 65% + 반열림(onset/offset) 35%, 깊이 목표 1~4cm rejection sampling.
#   비율은 유형별 평가 행(evaluate.py 시나리오)으로 검증 후 조정한다 — 하드코딩 금지.
# =====================================================================
DECLIP_MODE = True                            # False = 구(舊) 클린→클린 정규화 학습 (비교/재현용)
RUN_TAG_BASE = "tfm_declip_cov_l10.1"               # 'tfm' 접두어 → PVTVAE run 폴더와 절대 충돌 안 함
# [주의] 캡슐 반지름 시대 구분자를 '자동으로' 덧붙인다 (2026-08-12 해부학 반지름 도입).
#    반지름이 바뀌면 물리 손실 임계값·주입 깊이 목표·전 지표가 함께 바뀌므로, 태그를 손으로
#    붙이는 것을 잊으면 같은 λ의 구시대 폴더에서 resume해 실험이 조용히 오염된다.
#    RADII_TAG = "_anat"(신판) / ""(legacy) → dataset_pipeline.RADII_MODE 하나만 바꾸면 된다.
RUN_TAG = (RUN_TAG_BASE + RADII_TAG) if DECLIP_MODE else "tfm"
CORRUPTION_SEED = 777                         # 주입 파라미터 난수 시드 (run_config에 기록)
CORRUPTION_CFG = corruption.make_cfg(
    clean_ratio=0.5,        # 항등 보존 앵커 (모델이 클린 입력에 충돌을 '새로 만들던' 문제 방지)
    transient_ratio=0.3,    # 손상 샘플 중 일시적 비율 — 첫 run 30/70 합의
)

# 물리 손실 커리큘럼: 처음엔 동작만 배우고, 이후 선형 램프로 LAMBDA_PHYS까지 상승.
# warmup/ramp는 EPOCHS에 비례(각 10%)해 자동 산출된다: EPOCHS=100 -> 10/10, EPOCHS=200 -> 20/20.
# [주의] 이어 학습에서는 EPOCHS를 키우면 warmup 도 같이 커진다. warmup 이 재개 지점보다 크면
#   λ_phys 가 0으로 되돌아가 조용히 커리큘럼이 리셋된다 — 재개 드라이버(train_to2000.py)가
#   이 값을 고정하고 재개 에폭과 대조해 검사한다.
PHYS_WARMUP_EPOCHS = EPOCHS // 10   # 이 에폭까지는 물리 손실 0 (순수 동작 학습)
PHYS_RAMP_EPOCHS   = EPOCHS // 10   # warmup 이후 0 -> LAMBDA_PHYS 로 선형 상승하는 구간 길이

# ---------------------------------------------------------------------
# LR 스케줄 — [2026-09-22] EPOCHS 와 분리했다.
#   구판은 LR_STEP_SIZE = EPOCHS//4 였다. 이어 학습으로 EPOCHS 를 바꿀 때마다 반감 주기가 따라
#   움직여, 과거 구간의 LR 이력과 어긋나는 것이 문제였다 (1500 목표 때도 EPOCHS//4=375 를 쓰지 않고
#   500 으로 손으로 고정해야 했다 — claude_analysis/train_to2000.py 참조).
#   이제 반감 지점을 '에폭 번호'로 직접 적는다. EPOCHS 를 2500 → 3000 으로 올려도 스케줄은 불변이다.
#
#   LR_MILESTONES 로 만든 실제 LR (LEARNING_RATE=1e-4, gamma 0.5):
#     501~1000 : 5.0e-5     (기존 이력 그대로)
#     1001~2000: 2.5e-5     [2026-09-22 사용자 결정] 반감 주기를 500 → 1000 으로 늘려
#                            1501~2000 을 1001~1500 과 '같은 LR' 로 둔다. 수렴 판정의 블록 비교가
#                            LR 변화와 섞이지 않게 하기 위한 것 (같은 이유로 09-11 의 'LR 고갈' 함정을 피한다).
#     2001~3000: 1.25e-5
#     3001~4000: 6.25e-6 ...
#   LR_MILESTONES=None 으로 두면 구판 StepLR(LR_STEP_SIZE) 로 되돌아간다 (과거 런 재현용).
# ---------------------------------------------------------------------
LR_MILESTONES = [500, 1000, 2000, 3000, 4000, 5000]
LR_STEP_SIZE  = EPOCHS // 4         # LR_MILESTONES=None 일 때만 쓰이는 구판 StepLR 주기

# 전처리된 데이터/오프셋 경로 (새 양식: [Frames, 87] = Hips Pos 3 + Quats 84).
# 프로젝트 루트에서 실행하는 것이 표준이지만, AI_model/ 안에서 실행해도 동작하도록 폴백 포함.
PT_DIR = "processed_motions_VMC" if os.path.exists("processed_motions_VMC") else "../processed_motions_VMC"
OFFSET_CSV_PATH = "Sample_Data/Standard_BoneOffsets.csv" \
    if os.path.exists("Sample_Data/Standard_BoneOffsets.csv") \
    else "../Sample_Data/Standard_BoneOffsets.csv"

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
    print(f"🧪 아키텍처: TransformerDenoiser (결정론적, 프레임별 잠재 — 기본 아키텍처)")

    # 3. 모델, 데이터, 물리 엔진 초기화 (무거운 로딩은 import 시가 아니라 학습 실행 시에만)
    # split='train': held-out 평가 파일을 제외한 학습용 파일만 사용 (과적합 검증 가능하도록)
    dataset = BandaiMotionDataset(PT_DIR, seq_len=30, split='train')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # 윈도우 에러 방지용 0
    print(f"📚 학습 파일 수(train split): {len(dataset)}")

    model = TransformerDenoiser(input_dim=87, output_dim=84, latent_dim=64).to(DEVICE)
    physics_engine = DifferentiablePhysics(PARENTS, BONE_RADII, offset_csv_path=OFFSET_CSV_PATH).to(DEVICE)
    # 손상 주입의 깊이 검사용 CPU 물리 엔진 — 주입은 작은 텐서 연산이 많아 GPU 런치
    # 오버헤드가 오히려 손해이므로 CPU에서 수행 후 결과만 DEVICE로 올린다.
    physics_cpu = DifferentiablePhysics(PARENTS, BONE_RADII, offset_csv_path=OFFSET_CSV_PATH)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # LR 스케줄: LR_MILESTONES가 있으면 '에폭 번호로 고정된' 반감 지점을 쓰고(2026-09-22),
    # None이면 구판 StepLR(EPOCHS 비례 주기)로 되돌아간다. 상단 주석의 표가 실제 LR이다.
    scheduler = (optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(LR_MILESTONES), gamma=0.5)
                 if LR_MILESTONES else
                 optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.5))

    # 이번 실험의 손실 가중치를 이름으로 하는 전용 폴더에 저장한다.
    # 예: checkpoints/tfm_declip_recon1_phys0.4_kl0/  → lambda 스윕/아키텍처 간 서로 덮어쓰지 않음.
    # 'tfm_' 태그가 아키텍처를 구분하므로 PVTVAE_baseline(declip) 폴더와 절대 충돌하지 않는다.
    run_name = make_run_name(LAMBDA_RECON, LAMBDA_PHYS, BETA_KL, tag=RUN_TAG)
    run_dir = os.path.join(resolve_ckpt_root(), run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"📁 이번 실험 저장 폴더: {run_dir}")

    start_epoch = 1

    # 재개는 '같은 가중치' 폴더 안에서만 한다 (다른 lambda는 자기 폴더에서 새로 시작).
    # [주의] 체크포인트 파일명은 반드시 'pvtvae_epoch_*.pth' 를 유지한다 — 공유 헬퍼
    #    (find_latest_checkpoint_in / find_run_dir_by_config / list_available_runs)가
    #    이 패턴으로 글롭하므로, 다른 이름으로 저장하면 evaluate/inference/demo_maker가
    #    가중치를 찾지 못하고 재개 파싱(split('_')[2])도 깨진다.
    #    아키텍처 구분은 파일명이 아니라 '폴더명의 tfm_ 태그'가 담당한다.
    latest_ckpt = find_latest_checkpoint_in(run_dir)
    if latest_ckpt:
        try:
            model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))
            start_epoch = int(os.path.basename(latest_ckpt).split('_')[2].split('.')[0]) + 1
            print(f"🔄 이전 학습 상태 로드 완료: {os.path.basename(latest_ckpt)} (Epoch {start_epoch}부터 재시작)")
        except RuntimeError as e:
            print(f"⚠️ 기존 체크포인트({os.path.basename(latest_ckpt)})가 현재 아키텍처와 호환되지 않아 무시하고 처음부터 학습합니다.")
            print(f"   (원인: {str(e).splitlines()[0]})")

    # [2026-09-22 안전장치] 이어 학습에서 커리큘럼이 조용히 되감기는 것을 막는다.
    #   warmup/ramp 는 EPOCHS 비례라, 목표 에폭을 크게 올리면 재개 지점보다 커질 수 있다.
    #   그러면 이미 λ_phys 최대값으로 학습된 모델이 다시 물리 손실 0에서 출발한다 —
    #   로그만 봐서는 알아채기 어렵고, 지표는 몇백 에폭 뒤에야 틀어진다. 그래서 여기서 멈춘다.
    if start_epoch > 1 and PHYS_WARMUP_EPOCHS >= start_epoch:
        raise SystemExit(
            f"❌ 재개 지점(ep{start_epoch})이 물리 손실 warmup({PHYS_WARMUP_EPOCHS}) 안에 있습니다. "
            f"EPOCHS={EPOCHS} 에 비례해 warmup 이 커진 탓입니다.\n"
            f"   PHYS_WARMUP_EPOCHS / PHYS_RAMP_EPOCHS 를 직전 학습 값으로 고정한 뒤 다시 실행하세요 "
            f"(claude_analysis/train_to2000.py 가 --phys-warmup 으로 고정해 부릅니다).")

    if start_epoch > EPOCHS:
        raise SystemExit(f"❌ 이미 ep{start_epoch - 1} 까지 학습돼 목표 EPOCHS={EPOCHS} 를 넘었습니다. "
                         f"더 늘리려면 train.py 의 EPOCHS 를 올리거나 드라이버에 --epochs 를 주세요.")

    # 이번 학습에 사용된 실험용 가중치를 기록 (evaluate.py가 읽어 지표와 함께 기록)
    run_config = {
        "architecture": "transformer_denoiser",   # 기본 아키텍처 (VAE 제거)
        "lambda_recon": LAMBDA_RECON,
        "lambda_phys": LAMBDA_PHYS,
        "beta_kl": BETA_KL,                        # 0.0 — KL 손실 자체가 없음 (이름 호환용)
        "phys_warmup_epochs": PHYS_WARMUP_EPOCHS,
        "phys_ramp_epochs": PHYS_RAMP_EPOCHS,
        "epochs": EPOCHS,
        # LR 스케줄 (2026-09-22 추가) — 이 세 값이 없으면 어떤 LR 이력의 가중치인지 사후에 알 수 없다.
        "learning_rate": LEARNING_RATE,
        "lr_milestones": list(LR_MILESTONES) if LR_MILESTONES else None,
        "lr_step_size": None if LR_MILESTONES else LR_STEP_SIZE,
        "start_epoch": start_epoch,               # 이 실행이 이어받은 지점
        # §4.1 디클리핑 실험 재현에 필요한 전체 컨텍스트 (없으면 시대/조건 구분 불가)
        "run_tag": RUN_TAG,
        "declip_mode": DECLIP_MODE,
        "corruption_seed": CORRUPTION_SEED,
        "corruption": CORRUPTION_CFG if DECLIP_MODE else None,
        # 캡슐 반지름 시대 (2026-08-12~). 반지름은 물리 손실 임계값과 주입 깊이 목표를
        # 동시에 정하므로, 이 세 값이 다르면 다른 run과 수치를 비교할 수 없다.
        "radii_mode": RADII_MODE,
        "radii_kappa": SOFT_TISSUE_KAPPA,
        "radii_stature_m": SKELETON_STATURE_M,
        "fps": 30,          # Bandai 공개 데이터셋 공식 30fps (보고서 §7-A에서 확정)
        "seq_len": 30,      # 30프레임 @ 30fps = 1.0초 문맥 (v1 유지 합의)
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    log_file = os.path.join(run_dir, "log.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("🚀 Training Started (TransformerDenoiser)\n")
        f.write(f"[config] LAMBDA_RECON={LAMBDA_RECON} LAMBDA_PHYS={LAMBDA_PHYS} (no KL)\n")
        # 이어 학습 구간과 LR 스케줄을 로그에 남긴다 (2026-09-22) — log.txt 만 보고도
        # 어느 에폭 구간이 어떤 LR 로 돌았는지 사후 확인할 수 있게.
        f.write(f"[schedule] epochs {start_epoch}..{EPOCHS} | lr {LEARNING_RATE:g} "
                f"x0.5 @ {LR_MILESTONES if LR_MILESTONES else f'step {LR_STEP_SIZE}'} "
                f"| phys warmup {PHYS_WARMUP_EPOCHS} ramp {PHYS_RAMP_EPOCHS}\n")
        f.write("=" * 50 + "\n")

    # §4.1 손상 주입용 난수원 (파라미터 추첨 전용 — 텐서 연산 시드와 독립)
    corrupt_rng = random.Random(CORRUPTION_SEED)
    
    # 재개 시 LR 스케줄을 이미 지난 에폭 수만큼 전진시켜 일관성 유지
    for _ in range(start_epoch - 1):
        scheduler.step()

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_recon_loss = 0
        total_phys_loss = 0
        n_corrupted = 0     # 이번 에폭에 실제 주입된 샘플 수 (유형 무관)
        n_fallback = 0      # 지속 주입이 목표 깊이 달성에 실패해 클린으로 폴백한 수

        # 단계적 물리 제약 (Curriculum Learning)
        # 처음 PHYS_WARMUP_EPOCHS 동안은 동작만 배우고(물리 0),
        # 이후 PHYS_RAMP_EPOCHS 구간에 걸쳐 0 -> LAMBDA_PHYS 로 선형 상승 후 유지.
        if epoch <= PHYS_WARMUP_EPOCHS:
            lambda_phys = 0.0
        else:
            ramp = min(1.0, (epoch - PHYS_WARMUP_EPOCHS) / max(1, PHYS_RAMP_EPOCHS))
            lambda_phys = LAMBDA_PHYS * ramp

        # [주의] §4.1 핵심: 모델 '입력'에만 클리핑을 주입한다 (타겟은 클린 원본 유지).
        #    일시적 주입 → 시간 문맥 기반 복원 학습 / 지속적(작은 깊이) → 최소 사영 학습 /
        #    클린 절반 → "손상이 없으면 건드리지 마라" (항등 보존, R2 앵커).
        # 주입은 CPU 작업(배치당 ~수십 ms)이라 메인 루프에서 직접 부르면 그동안 GPU가 논다 →
        # corrupted_batches가 배치 N을 GPU가 학습하는 동안 배치 N+1의 주입을 백그라운드
        # 스레드로 준비한다 (주입 순서·난수 소비·결과는 직접 호출과 완전 동일 — corruption.py 참조).
        #    단, GPU 주입은 학습 스텝과 같은 스트림을 쓰므로 프리페치가 손해다(corruption.py 참조)
        #    → INJECT_ON_DEVICE 이면 배치마다 직접 부른다. 난수 소비 순서는 두 경로가 같다.
        if DECLIP_MODE and INJECT_ON_DEVICE and DEVICE != 'cpu':
            batch_iter = ((b, *corruption.corrupt_batch(
                              b.to(DEVICE), physics_engine, CORRUPTION_CFG, corrupt_rng, COLLIDING_PAIRS))
                          for b in dataloader)
        elif DECLIP_MODE:
            batch_iter = corruption.corrupted_batches(
                dataloader, physics_cpu, CORRUPTION_CFG, corrupt_rng, COLLIDING_PAIRS)
        else:
            batch_iter = ((b, None, None) for b in dataloader)   # 구(舊) 클린→클린 학습 재현용

        for batch_data, corrupted, metas in batch_iter:
            # batch_data: [Batch, 30, 87] (Hips Pos 3 + Quats 84) — '클린' 윈도우 = 항상 타겟
            if corrupted is not None:
                model_input = corrupted.to(DEVICE)
                n_corrupted += sum(1 for m in metas if m['type'] != 'clean')
                n_fallback += sum(1 for m in metas if m.get('fallback'))
            else:
                model_input = None
            clean_batch = batch_data.to(DEVICE)
            if model_input is None:
                model_input = clean_batch

            optimizer.zero_grad()

            # 1. Forward — 출력 [Batch, 30, 87] (Hips 통과 + 교정된 Quats).
            #    PVTVAE와 달리 반환값이 단일 텐서다 (mu/logvar 없음).
            recon_motion = model(model_input)

            # 위치/회전 분리 — 손실의 기준(타겟)은 클린 원본이다
            hips_pos     = clean_batch[..., :3]    # [B, 30, 3]  루트 위치 (주입은 회전만 변경)
            target_quats = clean_batch[..., 3:]    # [B, 30, 84] 클린 정답
            recon_quats  = recon_motion[..., 3:]   # [B, 30, 84]

            loss_recon_quat = nn.MSELoss()(recon_quats, target_quats)
            # [동결 v1 복원 2026-09-11] L1 희소성 항 (alpha=0.1). git b01c1fc 와 동일한 형태다.
            #   L1 은 '작은 오차'를 지배해 항등 근방에서 일정한 압력을 유지한다 — MSE-only 실험
            #   (alpha=0)은 두 lambda 모두에서 do-no-harm 을 깨뜨렸다. 이 항은 하중을 받는다.
            loss_sparsity = nn.L1Loss()(recon_quats, target_quats)

            loss_recon = loss_recon_quat + loss_sparsity * 0.1

            # 물리 엔진 연동: FK로 관절 월드 좌표 복원 → 충돌 Loss
            loss_phys = torch.tensor(0.0, device=DEVICE)
            if lambda_phys > 0:
                global_pos = physics_engine.forward_kinematics(hips_pos, recon_quats)
                loss_phys = physics_engine.get_collision_loss(global_pos, COLLIDING_PAIRS)

            # 최종 역전파 — KL 항이 없다는 것이 PVTVAE 학습과의 유일한 손실 차이
            loss = (LAMBDA_RECON * loss_recon) + (lambda_phys * loss_phys)
            loss.backward()

            # 안전벨트: 기울기(Gradient)가 폭발하지 않도록 최대치 제한
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_recon_loss += loss_recon.item()
            total_phys_loss += loss_phys.item()

        # 에폭 결과 출력
        num_batches = len(dataloader)
        log_msg = (f"Epoch [{epoch}/{EPOCHS}] "
                   f"Recon: {total_recon_loss/num_batches:.4f} (λ={LAMBDA_RECON:.3f}) | "
                   f"Phys: {total_phys_loss/num_batches:.4f} (λ={lambda_phys:.3f})")
        if DECLIP_MODE:
            # 주입 통계: 손상 샘플 수 / 지속-주입 깊이 실패 폴백 수 (폴백 급증 = 주입기 점검 신호)
            log_msg += f" | Inject: {n_corrupted} (fb {n_fallback})"

        # 터미널 화면에 출력
        print(log_msg)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + f" | LR: {current_lr:.6f}\n")

        # 모델 체크포인트 저장 (10 에폭마다) — 파일명 규칙은 상단 재개 로직의 주석 참조
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(run_dir, f"pvtvae_epoch_{epoch}.pth"))
            ckpt_msg = f"💾 Checkpoint saved: {run_name}/pvtvae_epoch_{epoch}.pth"
            print(ckpt_msg)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(ckpt_msg + "\n")


if __name__ == "__main__":
    train()
