"""확장 아키텍처 학습 — AI_model/train.py 의 구조를 그대로 따르되 모델만 교체한다.

[설계 원칙: 공정한 대조]
  손실·데이터·손상 주입·커리큘럼·옵티마이저·클리핑·저장 규칙을 **AI_model/train.py와
  문자 그대로 동일하게** 유지한다. 그래야 "아키텍처만 바꿨을 때 무엇이 달라지는가"를
  말할 수 있다. 대조군은 같은 손실 설정으로 이미 학습된 소형 런이다 (아래 BASELINE_RUN).

[이 파일이 AI_model 을 import 하는 이유]
  데이터 계약(dataset_pipeline) · 물리(physics_module) · 손상 주입(corruption)은
  **복사하면 안 된다** — 복사본은 조용히 갈라지고, 그 순간 두 실험은 비교 불가능해진다.
  따라서 sys.path에 AI_model 을 얹어 원본을 그대로 import 한다.
  **AI_model 의 어떤 파일도 수정하지 않는다.**

[체크포인트 폴더 — 가장 중요한 안전장치]
  run 폴더 이름은 (λ_recon, λ_phys, β_KL, RUN_TAG) 로만 만들어진다. 아키텍처가 달라도
  태그가 같으면 **같은 폴더에서 resume을 시도**하고, 구조가 다르면 state_dict 로드가
  실패해 조용히 처음부터 다시 학습하면서 **서로 다른 구조의 .pth 를 한 폴더에 섞는다.**
  그래서 RUN_TAG 에 models.arch_tag() 를 박는다 (예: ..._d256h8L4f1024_anat).
  추가로 run_config.json 의 아키텍처 필드를 읽어 불일치면 **즉시 중단**한다.

[파일 이름이 AI_model 과 다른 이유 — 반드시 알아야 할 함정]
  처음에는 `models.py` / `train.py` 로 파리티를 맞췄다가 즉시 되돌렸다.
  이 폴더를 실행하면 sys.path[0] 이 이 폴더가 되어 **AI_model 의 동명 모듈을 가린다.**
  그러면 AI_model/evaluate.py 내부의 `from models import TransformerDenoiserCompat` 이
  이 폴더의 models.py 를 잡아 ImportError 로 죽는다(그 이름이 여기 없으므로).
  `from train import ...` 도 마찬가지로 조용히 이 폴더의 상수를 읽는다.
  ⇒ 파일명을 `models_large.py` / `train_large.py` 로 분리해 **어떤 sys.path 순서에서도
     겹치지 않게** 만들었다. 역할 구성은 AI_model 과 1:1 로 같다.

실행:
    python AI_model_large/train_large.py    # 프로젝트 루트에서 (표준)
    python train_large.py                   # AI_model_large/ 안에서도 동작
"""

import os
import sys
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- AI_model 원본 모듈을 import 경로에 얹는다 (파일은 건드리지 않는다) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_AI_MODEL_DIR = os.path.join(os.path.dirname(_HERE), "AI_model")
if _AI_MODEL_DIR not in sys.path:
    sys.path.insert(0, _AI_MODEL_DIR)
# 자기 자신(AI_model_large)이 우선하도록 맨 앞에 둔다 — models 이름이 양쪽에 있기 때문.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset_pipeline import (BandaiMotionDataset, PARENTS, BONE_RADII, RADII_MODE, RADII_TAG,
                              SOFT_TISSUE_KAPPA, SKELETON_STATURE_M,
                              make_run_name, resolve_ckpt_root, find_latest_checkpoint_in)
from physics_module import DifferentiablePhysics
import corruption

import models_large as large_models
from models_large import TransformerDenoiserLarge

# 1. 하이퍼파라미터 및 환경 설정 — AI_model/train.py 와 동일한 값
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# [500 에폭 확장 2026-09-12] 소형 v1 을 500 까지 이어 학습한 결과(claude_analysis/
#   epoch_sweep_500_20260911.md)가 "100 에폭은 수렴이 아니라 LR 고갈"임을 보였으므로,
#   확장 아키텍처도 같은 예산에서 평가한다. 아래 커리큘럼 상수가 모두 EPOCHS 비례라
#   500 으로 두면 웜업 50 / 램프 50 / LR step 125 로 **스케줄이 그대로 5배 늘어난다**
#   (LR 반감 횟수는 3회로 동일 ⇒ 최종 LR 1.25e-5 도 100 에폭 런과 같다).
#   ⚠️ 따라서 large@500 을 소형 대조군(BASELINE_RUN, 100 에폭)과 직접 비교하면
#      아키텍처와 학습 예산이 함께 바뀐 비교가 된다. 소형도 500 으로 이어 학습해야
#      "아키텍처만 다른 A/B"가 복원된다.
EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# =====================================================================
# [실험용 조절 손잡이] recon vs phys 가중치 — 대조군과 반드시 같은 값을 쓴다.
#   현재 값은 AI_model/train.py 의 현재 설정과 동일하다:
#     LAMBDA_PHYS=0.3, MSE 배수 1.3, DECLIP_MODE=True, 손상 시드 777
#   ⇒ 짝이 되는 소형 대조군 체크포인트:
#        checkpoints/tfm_declip_cov_MSEOnly_1.3_anat_recon1_phys0.3_kl0/
#   이 짝이 성립해야 "아키텍처만 다른 A/B"가 된다. λ를 바꾸려면 소형도 같이 바꿔야 한다.
# =====================================================================
LAMBDA_RECON = 1.0
LAMBDA_PHYS  = 0.3
BETA_KL      = 0.0   # [주의] 손실 아님 — 폴더 이름/공유 코드 호환용 상수 (0.0 고정)

MSE_MULTIPLIER = 1.3   # AI_model/train.py:228 의 `loss_recon_quat * 1.3` 과 동일

BASELINE_RUN = "tfm_declip_cov_MSEOnly_1.3_anat_recon1_phys0.3_kl0"
"""같은 손실 설정으로 학습된 소형 대조군 폴더 이름 (비교 기준. 학습에는 쓰이지 않는다)."""

# =====================================================================
# RUN_TAG — 'tfmL' 접두어(Large) + 아키텍처 태그 + 반지름 시대 태그
#   tfm_  (소형) / tfmL_ (확장) / declip_ (PVTVAE) 가 폴더에서 절대 겹치지 않는다.
# =====================================================================
DECLIP_MODE = True
RUN_TAG_BASE = "tfmL_declip_cov_MSEOnly_1.3"
RUN_TAG = (RUN_TAG_BASE + large_models.arch_tag() + RADII_TAG) if DECLIP_MODE else "tfmL"

CORRUPTION_SEED = 777
CORRUPTION_CFG = corruption.make_cfg(
    clean_ratio=0.5,
    transient_ratio=0.3,
)

# 물리 손실 커리큘럼 — 원본과 동일한 비례식
PHYS_WARMUP_EPOCHS = EPOCHS // 10
PHYS_RAMP_EPOCHS   = EPOCHS // 10
LR_STEP_SIZE       = EPOCHS // 4

# 데이터/오프셋 경로 — 프로젝트 루트 실행이 표준, 하위 폴더 실행도 되도록 폴백.
_ROOT = os.path.dirname(_HERE)
PT_DIR = ("processed_motions_VMC" if os.path.exists("processed_motions_VMC")
          else os.path.join(_ROOT, "processed_motions_VMC"))
OFFSET_CSV_PATH = ("Sample_Data/Standard_BoneOffsets.csv"
                   if os.path.exists("Sample_Data/Standard_BoneOffsets.csv")
                   else os.path.join(_ROOT, "Sample_Data", "Standard_BoneOffsets.csv"))

# 2. 핵심 충돌 페어 — AI_model/train.py 와 동일해야 한다 (아래 _assert_pairs_match 가 검사)
COLLIDING_PAIRS = [
    (('Hips', 'Chest'), ('LeftLowerArm', 'LeftHand')),   # 몸통 vs 왼팔
    (('Hips', 'Chest'), ('RightLowerArm', 'RightHand')), # 몸통 vs 오른팔
    (('LeftLowerArm', 'LeftHand'), ('RightLowerArm', 'RightHand')), # 왼팔 vs 오른팔
    (('LeftLowerLeg', 'LeftFoot'), ('RightLowerLeg', 'RightFoot'))  # 왼다리 vs 오른다리
]


def _assert_pairs_match():
    """소형 train.py 의 COLLIDING_PAIRS 와 다르면 즉시 중단.

    페어가 갈리면 물리 손실의 대상이 달라져 A/B 가 성립하지 않는다. import 는 이 함수
    안에서만 한다 — 학습 런타임에 원본 train.py 의 부작용(데이터 경로 탐색)을 끌어들이지
    않기 위해서다.
    """
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_ai_model_train", os.path.join(_AI_MODEL_DIR, "train.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:                      # 원본을 읽을 수 없으면 경고만 (학습은 계속)
        print(f"⚠️ 대조군 train.py 를 읽지 못해 페어 일치 검사를 건너뜁니다: {e}")
        return
    if list(mod.COLLIDING_PAIRS) != list(COLLIDING_PAIRS):
        raise SystemExit(
            "❌ COLLIDING_PAIRS 가 AI_model/train.py 와 다릅니다. 물리 손실 대상이 달라져\n"
            "   아키텍처 A/B 가 성립하지 않습니다. 두 파일을 일치시키세요.")


def _check_run_dir_arch(run_dir, arch):
    """폴더에 남아 있는 run_config.json 의 아키텍처가 지금 설정과 다르면 중단.

    arch_tag 덕에 보통은 폴더가 갈리지만, 사용자가 태그를 직접 손댔을 때의 마지막 방어선이다.
    """
    cfg_path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        return
    try:
        with open(cfg_path, encoding="utf-8") as f:
            old = json.load(f)
    except Exception:
        return
    keys = ("d_model", "nhead", "num_layers", "dim_feedforward")
    old_arch = {k: old.get(k) for k in keys}
    new_arch = {k: arch[k] for k in keys}
    if any(v is not None for v in old_arch.values()) and old_arch != new_arch:
        raise SystemExit(
            f"❌ 같은 폴더에 다른 아키텍처의 학습 기록이 있습니다:\n"
            f"   폴더: {run_dir}\n"
            f"   기존: {old_arch}\n"
            f"   현재: {new_arch}\n"
            f"   RUN_TAG 를 바꾸거나 폴더를 옮기세요 (구조가 섞인 체크포인트는 복구 불가).")


# 4. 메인 학습 루프 — AI_model/train.py 와 같은 순서/같은 로그
def train():
    print(f"🔥 학습 디바이스: {DEVICE}")
    print(f"🧪 아키텍처: TransformerDenoiserLarge "
          f"(d_model={large_models.LARGE_D_MODEL}, nhead={large_models.LARGE_NHEAD}, "
          f"layers={large_models.LARGE_NUM_LAYERS}, ff={large_models.LARGE_DIM_FEEDFORWARD})")

    _assert_pairs_match()

    dataset = BandaiMotionDataset(PT_DIR, seq_len=30, split='train')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"📚 학습 파일 수(train split): {len(dataset)}")

    model = TransformerDenoiserLarge(input_dim=87, output_dim=84).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧮 파라미터 수: {n_params:,}")

    physics_engine = DifferentiablePhysics(PARENTS, BONE_RADII, offset_csv_path=OFFSET_CSV_PATH).to(DEVICE)
    physics_cpu = DifferentiablePhysics(PARENTS, BONE_RADII, offset_csv_path=OFFSET_CSV_PATH)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.5)

    run_name = make_run_name(LAMBDA_RECON, LAMBDA_PHYS, BETA_KL, tag=RUN_TAG)
    run_dir = os.path.join(resolve_ckpt_root(), run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"📁 이번 실험 저장 폴더: {run_dir}")
    print(f"⚖️  대조군(소형, 같은 손실): checkpoints/{BASELINE_RUN}")

    _check_run_dir_arch(run_dir, model.arch)

    start_epoch = 1

    # 재개는 '같은 폴더' 안에서만. 파일명 규칙 'pvtvae_epoch_*.pth' 는 공유 헬퍼가
    # 글롭하는 패턴이므로 절대 바꾸지 않는다 (아키텍처 구분은 폴더명이 담당).
    latest_ckpt = find_latest_checkpoint_in(run_dir)
    if latest_ckpt:
        try:
            model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))
            start_epoch = int(os.path.basename(latest_ckpt).split('_')[2].split('.')[0]) + 1
            print(f"🔄 이전 학습 상태 로드 완료: {os.path.basename(latest_ckpt)} (Epoch {start_epoch}부터 재시작)")
        except RuntimeError as e:
            raise SystemExit(
                f"❌ 폴더에 있는 체크포인트({os.path.basename(latest_ckpt)})가 현재 구조와 호환되지 않습니다.\n"
                f"   원본 train.py 는 이 경우 조용히 처음부터 학습하지만, 그러면 서로 다른 구조의\n"
                f"   .pth 가 한 폴더에 섞입니다. 여기서는 중단합니다 — 폴더를 비우거나 RUN_TAG 를 바꾸세요.\n"
                f"   (원인: {str(e).splitlines()[0]})")

    run_config = {
        "architecture": "transformer_denoiser_large",
        "d_model": model.arch["d_model"],
        "nhead": model.arch["nhead"],
        "num_layers": model.arch["num_layers"],
        "dim_feedforward": model.arch["dim_feedforward"],
        "dropout": model.arch["dropout"],
        "latent_dim": model.arch["latent_dim"],
        "n_params": n_params,
        "baseline_run": BASELINE_RUN,
        "mse_multiplier": MSE_MULTIPLIER,
        "lambda_recon": LAMBDA_RECON,
        "lambda_phys": LAMBDA_PHYS,
        "beta_kl": BETA_KL,
        "phys_warmup_epochs": PHYS_WARMUP_EPOCHS,
        "phys_ramp_epochs": PHYS_RAMP_EPOCHS,
        "epochs": EPOCHS,
        "run_tag": RUN_TAG,
        "declip_mode": DECLIP_MODE,
        "corruption_seed": CORRUPTION_SEED,
        "corruption": CORRUPTION_CFG if DECLIP_MODE else None,
        "radii_mode": RADII_MODE,
        "radii_kappa": SOFT_TISSUE_KAPPA,
        "radii_stature_m": SKELETON_STATURE_M,
        "fps": 30,
        "seq_len": 30,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    log_file = os.path.join(run_dir, "log.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("🚀 Training Started (TransformerDenoiserLarge)\n")
        f.write(f"[config] LAMBDA_RECON={LAMBDA_RECON} LAMBDA_PHYS={LAMBDA_PHYS} (no KL) "
                f"arch={large_models.arch_tag()} params={n_params}\n")
        f.write("=" * 50 + "\n")

    corrupt_rng = random.Random(CORRUPTION_SEED)

    for _ in range(start_epoch - 1):
        scheduler.step()

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_recon_loss = 0
        total_phys_loss = 0
        n_corrupted = 0
        n_fallback = 0

        if epoch <= PHYS_WARMUP_EPOCHS:
            lambda_phys = 0.0
        else:
            ramp = min(1.0, (epoch - PHYS_WARMUP_EPOCHS) / max(1, PHYS_RAMP_EPOCHS))
            lambda_phys = LAMBDA_PHYS * ramp

        if DECLIP_MODE:
            batch_iter = corruption.corrupted_batches(
                dataloader, physics_cpu, CORRUPTION_CFG, corrupt_rng, COLLIDING_PAIRS)
        else:
            batch_iter = ((b, None, None) for b in dataloader)

        for batch_data, corrupted, metas in batch_iter:
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

            recon_motion = model(model_input)

            hips_pos     = clean_batch[..., :3]
            target_quats = clean_batch[..., 3:]
            recon_quats  = recon_motion[..., 3:]

            loss_recon_quat = nn.MSELoss()(recon_quats, target_quats)
            loss_recon = loss_recon_quat * MSE_MULTIPLIER

            loss_phys = torch.tensor(0.0, device=DEVICE)
            if lambda_phys > 0:
                global_pos = physics_engine.forward_kinematics(hips_pos, recon_quats)
                loss_phys = physics_engine.get_collision_loss(global_pos, COLLIDING_PAIRS)

            loss = (LAMBDA_RECON * loss_recon) + (lambda_phys * loss_phys)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_recon_loss += loss_recon.item()
            total_phys_loss += loss_phys.item()

        num_batches = len(dataloader)
        log_msg = (f"Epoch [{epoch}/{EPOCHS}] "
                   f"Recon: {total_recon_loss/num_batches:.4f} (λ={LAMBDA_RECON:.3f}) | "
                   f"Phys: {total_phys_loss/num_batches:.4f} (λ={lambda_phys:.3f})")
        if DECLIP_MODE:
            log_msg += f" | Inject: {n_corrupted} (fb {n_fallback})"

        print(log_msg)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + f" | LR: {current_lr:.6f}\n")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(run_dir, f"pvtvae_epoch_{epoch}.pth"))
            ckpt_msg = f"💾 Checkpoint saved: {run_name}/pvtvae_epoch_{epoch}.pth"
            print(ckpt_msg)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(ckpt_msg + "\n")


if __name__ == "__main__":
    train()
