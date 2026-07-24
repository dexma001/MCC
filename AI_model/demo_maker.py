"""
발표용 데모 생성기 (§4.1 디클리핑 시대판).

학습(train.py)·평가(evaluate.py)와 동일한 corruption.py 주입기로 held-out 테스트
파일에 클리핑을 주입하고, train.py 설정(LAMBDA_*/RUN_TAG)과 일치하는 run 폴더의
체크포인트로 교정한 결과를 demo_results/ 에 저장한다.
→ 이후 'python AI_model/dataset_pipeline.py' (VISUALIZE=COMPARE)로 시각화.

[구판과의 차이 — §4.1 반영]
  - 주입: 구판의 'LeftUpperArm 로컬 +Z 90° 고정' 주입은 §4.1 학습 분포에서
    의도적으로 제외된 계열(오염 검사용 legacy80의 초과판)이라 디클리핑 모델의
    최악 OOD 사례만 보여줬다 (실측 2026-07-09: 침투 7.25 → 9.2cm 악화).
    이제 학습/평가와 동일한 corruption.py 주입기(transient/persistent)를 쓴다.
  - 체크포인트: 하드코딩 경로 대신 evaluate.py와 동일한 설정 기반 선택
    (find_run_dir_by_config + RUN_TAG). train.py의 람다를 바꾸면 그 실험이 데모된다.
  - 소스 파일: 전체 파일이 아니라 held-out 테스트 분할에서만 추첨한다
    (학습에 본 적 없는 데이터라는 정직한 데모).
  - 주입이 충돌을 만들지 못하면 다른 파일로 재추첨한다 (2026-07-06 실증:
    같은 회전도 포즈에 따라 충돌 0이 될 수 있음 → rejection sampling 필수).
"""
import os
import json
import datetime
import random
import torch

# 기본 아키텍처(TransformerDenoiser) — Compat 어댑터로 (out, mu, logvar) 3-튜플 시그니처 유지.
from models import TransformerDenoiserCompat
from dataset_pipeline import (PARENTS, BONE_RADII, get_split_files,
                              make_run_name, find_run_dir_by_config,
                              find_latest_checkpoint_in, list_available_runs)
from physics_module import DifferentiablePhysics
import corruption
# 데모 대상 실험은 train.py의 설정으로 선택 (evaluate.py와 동일한 방식 — mtime 아님)
from train import LAMBDA_RECON, LAMBDA_PHYS, BETA_KL, RUN_TAG, COLLIDING_PAIRS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 30

# =====================================================================
# 데모 파라미터 — demo_results/demo_meta.json 에 기록되어 동일 데모 재생성 가능
# =====================================================================
DEMO_SCENARIO = 'persistent'    # 'transient'(글리치 복원) 또는 'persistent'(최소 사영)
DEMO_SEED = random.randrange(0, 4000)           # 파일/주입 추첨 시드 — 바꾸면 다른 데모가 나온다
TARGET_FILE = ""               # 지정 시 해당 .pt 파일 고정 (재현용), 빈 문자열 = 추첨
DEMO_MIN_DEPTH_CM = 2.0        # 화면에서 잘 보이는 최소 주입 깊이 — 미달 시 파일 재추첨
MAX_FILE_TRIES = 30            # 재추첨 상한 (초과 시 그때까지 최선의 후보 사용)

# 주입 '형태'는 학습/평가와 동일한 설계 기본값. 단 transient의 최소 깊이 하한만
# 데모 가시성 기준(DEMO_MIN_DEPTH_CM)으로 올린다 — 주입기 내부 재추첨이 깊은
# 충돌을 우선 채택하게 될 뿐, 각도/길이 분포(U[15°,70°], 5~20프레임)는 그대로다.
DEMO_CFG = corruption.make_cfg(transient_min_depth_cm=max(0.3, DEMO_MIN_DEPTH_CM))


def create_demo():
    print(f"[발표용 데모 생성기] §4.1 디클리핑 데모 — scenario='{DEMO_SCENARIO}'")

    # 1. 체크포인트: train.py 설정과 일치하는 run 폴더에서 최신 epoch 가중치 로드
    run_dir = find_run_dir_by_config(LAMBDA_RECON, LAMBDA_PHYS, BETA_KL, tag=RUN_TAG)
    if run_dir is None:
        target = make_run_name(LAMBDA_RECON, LAMBDA_PHYS, BETA_KL, tag=RUN_TAG)
        print(f"❌ 설정과 일치하는 학습 폴더(checkpoints/{target}/)를 찾을 수 없습니다.")
        avail = list_available_runs()
        if avail:
            print(f"   사용 가능한 실험 폴더: {avail}")
            print("   → train.py의 LAMBDA_*/DECLIP_MODE를 위 이름 중 하나에 맞춰 다시 실행하세요.")
        return
    ckpt_path = find_latest_checkpoint_in(run_dir)

    model = TransformerDenoiserCompat(input_dim=87, output_dim=84, latent_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"✅ 모델 로드: {os.path.relpath(ckpt_path)}")

    physics = DifferentiablePhysics(PARENTS, BONE_RADII)

    def depth_report(motion_87):
        """프레임별 최대 침투 깊이(cm) → (최대 깊이 cm, 충돌 프레임 수). 시각화와 동일 4페어."""
        dep = physics.get_penetration_depths_from_quats(
            motion_87[:, :3], motion_87[:, 3:], COLLIDING_PAIRS) * 100.0   # [F, P] cm
        fmax = dep.max(dim=1).values
        return float(fmax.max()), int((fmax > 1e-4).sum())

    # 2. 소스 모션: held-out 테스트 분할에서 추첨, 주입이 목표 깊이의 충돌을 만들 때까지
    #    파일을 재추첨한다 (rejection sampling — 주입은 포즈에 따라 무충돌일 수 있음).
    motions_dir = "processed_motions_VMC" if os.path.exists("processed_motions_VMC") \
        else "../processed_motions_VMC"
    inject_fn = corruption.inject_transient if DEMO_SCENARIO == 'transient' \
        else corruption.inject_persistent
    min_depth = DEMO_MIN_DEPTH_CM if DEMO_SCENARIO == 'transient' else 1.0

    if TARGET_FILE:
        cands = [TARGET_FILE]
    else:
        cands = get_split_files(motions_dir, split='test')
        random.Random(DEMO_SEED).shuffle(cands)

    chosen = None   # (파일, 클린 원본, 손상본, 주입 meta)
    best = None     # 전 후보 실패 시의 차선 (가장 깊은 충돌)
    for try_i, fpath in enumerate(cands[:MAX_FILE_TRIES]):
        full = torch.load(fpath)
        if full.shape[0] < SEQ_LEN:
            continue
        clean = full[:SEQ_LEN].clone()                                # [30, 87]
        rng = random.Random(DEMO_SEED + 10007 * try_i)
        corrupted, meta = inject_fn(clean, physics, DEMO_CFG, rng, COLLIDING_PAIRS)
        depth = meta.get('max_depth_cm', 0.0)
        if meta.get('collided') and depth >= min_depth:
            chosen = (fpath, clean, corrupted, meta)
            break
        if meta.get('collided') and (best is None or depth > best[3]['max_depth_cm']):
            best = (fpath, clean, corrupted, meta)
    if chosen is None:
        if best is None:
            print(f"❌ {MAX_FILE_TRIES}개 파일 모두 주입이 충돌을 만들지 못했습니다. "
                  f"DEMO_SEED를 바꿔 다시 시도하세요.")
            return
        chosen = best
        print(f"⚠️ 목표 깊이 {min_depth}cm 이상 실패 — 가장 깊은 후보로 진행합니다.")
    target_file, _clean_motion, demo_motion, inject_meta = chosen
    print(f"소스 파일(held-out): {target_file}")
    print(f"주입: {inject_meta['type']} | bone={inject_meta['bone']} | "
          f"frames={inject_meta['frames']} | depth={inject_meta['max_depth_cm']:.2f} cm")

    maxpen_b, ncoll_b = depth_report(demo_motion)
    print(f"주입된 충돌(Before): 최대 침투 {maxpen_b:.2f} cm | 충돌 프레임 {ncoll_b}/{demo_motion.shape[0]}")

    # 3. AI 교정 (Inference)
    with torch.no_grad():
        recon_motion, _, _ = model(demo_motion.unsqueeze(0).to(DEVICE))
    corrected_motion = recon_motion.squeeze(0).cpu()                  # [30, 87]

    maxpen_a, ncoll_a = depth_report(corrected_motion)
    print(f"교정 후 충돌(After) : 최대 침투 {maxpen_a:.2f} cm | 충돌 프레임 {ncoll_a}/{corrected_motion.shape[0]}")

    # 4. 시각화 툴이 읽도록 저장 ([30, 87]) — Before = 손상 입력, After = 모델 출력
    results_dir = "demo_results" if os.path.exists("processed_motions_VMC") else "../demo_results"
    os.makedirs(results_dir, exist_ok=True)
    torch.save(demo_motion, os.path.join(results_dir, "sample_original.pt"))
    torch.save(corrected_motion, os.path.join(results_dir, "sample_corrected.pt"))

    # 4-1. 데모 재현용 메타데이터: 어떤 파일/가중치/주입으로 만든 데모인지 기록
    #      (잘 나온 데모는 TARGET_FILE + DEMO_SEED 고정으로 그대로 재생성 가능)
    meta = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": str(target_file),
        "source_split": "test (held-out)" if not TARGET_FILE else "manual",
        "checkpoint": ckpt_path,
        "seq_len": SEQ_LEN,
        "demo_scenario": DEMO_SCENARIO,
        "demo_seed": DEMO_SEED,
        "inject": inject_meta,
        "max_pen_before_cm": round(maxpen_b, 2),
        "max_pen_after_cm": round(maxpen_a, 2),
        "collision_frames_before": ncoll_b,
        "collision_frames_after": ncoll_a,
    }
    with open(os.path.join(results_dir, "demo_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("준비 완료! 이제 'python AI_model/dataset_pipeline.py' (VISUALIZE=COMPARE)로 결과를 확인하세요.")


if __name__ == "__main__":
    create_demo()
