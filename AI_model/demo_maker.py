"""
발표용 데모 생성기 (§4.1 디클리핑 시대판).

학습(train.py)·평가(evaluate.py)와 동일한 corruption.py 주입기로 held-out 테스트
파일에 클리핑을 주입하고, train.py 설정(LAMBDA_*/RUN_TAG)과 일치하는 run 폴더의
체크포인트로 교정한 결과를 demo_results/ 에 저장한다.
→ 이후 'python AI_model/viz_motion.py compare' 로 시각화.

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

[무결성 보강 — 2026-08-07]
  - DEMO_SCENARIO는 DEMO_SCENARIOS 화이트리스트로 검증한다. 구판의 이항 if/else는
    오타('Persistent' 등)를 조용히 persistent로 처리하고 그 잘못된 문자열을
    demo_meta.json에 기록해 재현 기록을 거짓으로 만들었다.
  - 후보 채택/차선 폴백은 meta['collided']가 아니라 '깊이'로 판정한다. collided는
    '충돌 여부'가 아니라 '주입기 내부 임계 통과 여부'라서, 구판에서는 차선 폴백이
    도달 불가능한 죽은 코드였고 1.9cm로 실제 충돌한 데모가 "충돌 실패"로 보고됐다.
  - DEMO_MIN_DEPTH_CM이 두 시나리오 모두에 적용된다 (구판은 persistent일 때
    하드코딩 1.0으로 대체되어 무력했다). 지속형은 주입기 상한 4cm로 clamp.
  - TARGET_FILE 사용 시 held-out 여부를 실제로 확인해 화면과 meta에 정직하게 표기한다.
  - DEMO_SEED는 환경변수로 덮어쓸 수 있다 → 잘 나온 데모를 소스 수정 없이 재현.
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
# [R1.5-4 / R1.5-3] 후처리 레이어. evaluate.py와 '같은 모듈, 같은 순서'를 쓰는 것이 핵심이다
#   — 데모가 평가와 다른 파이프라인을 보여주면 발표 수치와 화면이 어긋난다.
import lowpass
import projection
# 데모 대상 실험은 train.py의 설정으로 선택 (evaluate.py와 동일한 방식 — mtime 아님)
from train import LAMBDA_RECON, LAMBDA_PHYS, BETA_KL, RUN_TAG, COLLIDING_PAIRS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 30

# =====================================================================
# 데모 파라미터 — demo_results/demo_meta.json 에 기록되어 동일 데모 재생성 가능
# =====================================================================
DEMO_SCENARIO = 'persistent'    # 'transient'(글리치 복원) 또는 'persistent'(최소 사영)

# 시나리오별 (주입 함수, 채택 깊이 하한). evaluate.py의 시나리오 중 이 둘만 데모가 지원한다
# ('clean'은 보여줄 충돌이 없고, 'legacy80'은 §4.1 학습 분포에서 의도적으로 제외된 계열).
# [주의] persistent 하한은 주입기의 목표 범위 상한(persistent_depth_range_cm[1]=4.0cm)을
#    넘길 수 없다 — 넘기면 어떤 후보도 채택되지 못해 영구 실패한다. 아래에서 clamp한다.
DEMO_SCENARIOS = {
    'transient':  corruption.inject_transient,
    'persistent': corruption.inject_persistent,
}
DEMO_SEED = random.randrange(0, 4000)           # 파일/주입 추첨 시드 — 바꾸면 다른 데모가 나온다
DEMO_SEED = int(os.environ.get("DEMO_SEED", DEMO_SEED))   # 재현: DEMO_SEED=1234 로 고정 실행 가능
# 지정 시 해당 .pt 파일 고정 (재현용), 빈 문자열 = 추첨.
# 현재 값: viz_inject.py가 seed=1234로 뽑은 파일과 동일 — 주입 시각화 gif와 데모가
# 같은 모션을 쓰도록 맞춘 것이다 (DEMO_SEED=1234와 함께 쓰면 주입 결과까지 일치).
TARGET_FILE = "" # processed_motions_VMC/dataset-1_call_normal_001.pt
DEMO_MIN_DEPTH_CM = 2.0        # 화면에서 잘 보이는 최소 주입 깊이 — 두 시나리오 모두에 적용
MAX_FILE_TRIES = 30            # 재추첨 상한 (초과 시 그때까지 최선의 후보 사용)

# =====================================================================
# [후처리 파이프라인] 모델 → 저역통과 필터 → 사영. 이 순서는 취향이 아니라 정확성 조건이다.
#   필터를 사영 '뒤'에 두면 저역통과가 사영 결과를 뭉개 관통을 되살린다
#   (V4 실측 2026-08-26, held-out 20파일: 4시나리오 전부 0.000cm → 4.58~5.11cm).
#   evaluate.py:_eval_one_scenario 와 동일한 순서이며, 바꾸지 말 것.
# [주의] 아래 두 스위치는 lowpass.LP_ENABLED / projection.PROJ_ENABLED 와 별개다.
#   평가 기본값(LP_ENABLED=False)은 회귀 게이트를 성립시키기 위한 것이고, 데모는
#   '완성된 파이프라인을 보여주는' 경로이므로 여기서 독립적으로 켠다. 실제로 어떤 구성이
#   쓰였는지는 demo_meta.json의 pipeline 항목에 기록된다.
# =====================================================================
DEMO_LOWPASS = True            # 모델 출력에 저역통과 필터 적용 (lowpass.LP_MODE/LP_WINDOW 사용)
DEMO_PROJECTION = True         # 필터 뒤에 사영 적용 (projection.PROJ_K/PROJ_OMEGA 사용)

# 결과 애니메이션 저장 — viz_motion의 COMPARE 뷰어를 그대로 재사용한다(렌더링 코드 중복 금지).
DEMO_SAVE_GIF = True           # demo_results/<DEMO_GIF_NAME> 으로 gif 저장
DEMO_GIF_NAME = "compare.gif"
DEMO_SHOW_ANIMATION = False    # True면 gif 저장 후 재생 창까지 띄운다(창을 닫아야 스크립트가 끝난다).
                               # 지터/뭉개짐은 정지 그림으로 판단할 수 없으므로, 눈으로 볼 때는
                               # 이 값을 True로 두거나 viz_motion.py compare 를 따로 실행한다.

# 주입 '형태'는 학습/평가와 동일한 설계 기본값. transient만 주입기 '내부' 재추첨 하한을
# 데모 가시성 기준으로 올린다 (각도/길이 분포 U[15°,70°]·5~20프레임은 그대로).
DEMO_CFG = corruption.make_cfg(transient_min_depth_cm=max(0.3, DEMO_MIN_DEPTH_CM))


def save_compare_gif(results_dir, gif_name=DEMO_GIF_NAME, show=DEMO_SHOW_ANIMATION):
    """demo_results/의 Before/After를 viz_motion의 COMPARE 뷰어로 렌더링해 gif로 저장한다.

    [주의] viz_motion.py 는 의도적으로 GitHub에서 제외된 로컬 전용 파일이다(.gitignore).
       따라서 여기서는 최상단 import가 아니라 '지연 import + 명시적 안내'로 처리한다 —
       새로 클론한 환경에서 데모 생성 자체가 ImportError로 죽으면 안 되기 때문이다.
       렌더링 코드를 여기에 복제하지 않는 이유도 같다: 캡슐/침투 색 규칙이 두 벌이 되면
       화면과 수치가 갈라진다 (viz_motion의 probe가 물리 엔진과 같은 기준을 쓴다).
    """
    gif_path = os.path.join(results_dir, gif_name)
    try:
        import viz_motion
    except ImportError as e:
        print(f"⚠️ gif 저장을 건너뜁니다 — viz_motion.py를 불러올 수 없습니다 ({e}).")
        print("   (viz_motion.py는 로컬 전용 파일입니다. 저장 없이 결과 .pt만 사용하세요.)")
        return None
    print(f"🎞️ 애니메이션 렌더링 중 → {gif_path}")
    viz_motion.visualize_compare(results_dir=results_dir, save_path=gif_path, show=show)
    return gif_path


def create_demo():
    # 오타로 인한 '조용한 persistent 데모'를 차단한다. 구판은 이항 if/else였기 때문에
    # 'Persistent'/'transiant' 같은 오타가 예외 없이 지속형으로 떨어지고, 그 잘못된
    # 문자열이 demo_meta.json에 그대로 기록되어 재현 기록이 거짓이 되었다.
    # (evaluate.py:390의 화이트리스트 검증과 동일한 정책)
    if DEMO_SCENARIO not in DEMO_SCENARIOS:
        raise ValueError(f"알 수 없는 시나리오: {DEMO_SCENARIO!r} "
                         f"(지원: {sorted(DEMO_SCENARIOS)})")
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

    def jitter_report(motion_87):
        """관절 위치 가속도 크기의 평균 (cm/frame^2). evaluate.calculate_acceleration_jitter와
        같은 [1,-2,1] 시간축 2차 차분이며, 저역통과 필터가 실제로 무엇을 줄였는지 보여준다.
        (evaluate를 import하지 않는 이유: 데모가 평가 모듈의 전역 설정에 묶이지 않게 한다.)"""
        gp = physics.compute_global_pos_tensor(motion_87[:, :3], motion_87[:, 3:])
        if gp.shape[0] < 3:
            return 0.0
        accel = gp[2:] - 2 * gp[1:-1] + gp[:-2]
        return float(torch.norm(accel, dim=-1).mean()) * 100.0

    # 2. 소스 모션: held-out 테스트 분할에서 추첨, 주입이 목표 깊이의 충돌을 만들 때까지
    #    파일을 재추첨한다 (rejection sampling — 주입은 포즈에 따라 무충돌일 수 있음).
    motions_dir = "processed_motions_VMC" if os.path.exists("processed_motions_VMC") \
        else "../processed_motions_VMC"
    inject_fn = DEMO_SCENARIOS[DEMO_SCENARIO]
    # 채택 하한은 두 시나리오 모두 DEMO_MIN_DEPTH_CM(가시성 기준). 단 persistent는 주입기가
    # 깊이를 [1,4]cm로 통제하므로 그 상한을 넘는 요구는 달성 불가 → 4cm로 clamp한다.
    min_depth = DEMO_MIN_DEPTH_CM
    if DEMO_SCENARIO == 'persistent':
        pmax = DEMO_CFG['persistent_depth_range_cm'][1]
        if min_depth > pmax:
            print(f"⚠️ DEMO_MIN_DEPTH_CM={DEMO_MIN_DEPTH_CM}cm는 지속형 주입 상한({pmax}cm) 초과 "
                  f"→ {pmax}cm로 낮춰 진행합니다.")
            min_depth = pmax

    test_files = get_split_files(motions_dir, split='test')
    if TARGET_FILE:
        cands = [TARGET_FILE]
        # TARGET_FILE은 분할을 우회하므로 held-out 여부를 직접 확인한다. 이 검증이 없으면
        # 학습에 쓴 파일로 데모를 만들고도 화면에는 '(held-out)'이 찍혀 발표가 부정직해진다.
        is_heldout = os.path.abspath(TARGET_FILE) in {os.path.abspath(p) for p in test_files}
        if not is_heldout:
            print(f"⚠️ TARGET_FILE이 held-out 테스트 분할에 없습니다 — 학습에 사용됐을 수 있습니다.")
    else:
        is_heldout = True
        cands = list(test_files)
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
        # [주의] meta['collided']는 '충돌했는가'가 아니라 '주입기 내부 임계를 통과했는가'다
        #    (corruption.py:224 / 342). transient는 임계 미달이면 실제로 관통했어도
        #    collided=False가 되므로, 차선 후보 추적은 collided가 아니라 depth로 한다.
        #    이 구분이 없으면 아래 best 폴백이 영원히 도달 불가능한 죽은 코드가 되고,
        #    1.9cm로 '충돌한' 데모가 "충돌을 만들지 못했습니다"로 잘못 보고된다.
        if depth >= min_depth:
            chosen = (fpath, clean, corrupted, meta)
            break
        if depth > 0.0 and (best is None or depth > best[3].get('max_depth_cm', 0.0)):
            best = (fpath, clean, corrupted, meta)
    if chosen is None:
        if best is None:
            print(f"❌ {MAX_FILE_TRIES}개 파일 모두 주입이 충돌을 만들지 못했습니다. "
                  f"DEMO_SEED를 바꿔 다시 시도하세요.")
            return
        chosen = best
        print(f"⚠️ 목표 깊이 {min_depth}cm 이상 실패 — 가장 깊은 후보"
              f"({best[3].get('max_depth_cm', 0.0):.2f}cm)로 진행합니다.")
    target_file, _clean_motion, demo_motion, inject_meta = chosen
    split_label = "held-out" if is_heldout else "⚠️ 분할 외(학습 데이터일 수 있음)"
    print(f"소스 파일({split_label}): {target_file}")
    print(f"주입: {inject_meta['type']} | bone={inject_meta['bone']} | "
          f"frames={inject_meta['frames']} | depth={inject_meta['max_depth_cm']:.2f} cm")

    maxpen_b, ncoll_b = depth_report(demo_motion)
    print(f"주입된 충돌(Before): 최대 침투 {maxpen_b:.2f} cm | 충돌 프레임 {ncoll_b}/{demo_motion.shape[0]}")

    # 3. AI 교정 (Inference) — 모델 → 저역통과 필터 → 사영
    with torch.no_grad():
        recon_motion, _, _ = model(demo_motion.unsqueeze(0).to(DEVICE))
    corrected_motion = recon_motion.squeeze(0).cpu()                  # [30, 87]

    # 단계별 수치를 전부 남긴다 — 어느 레이어가 무엇을 했는지 화면에서 바로 보이도록.
    # (필터는 지터를, 사영은 관통을 담당하므로 두 축을 함께 찍어야 역할이 드러난다.)
    stage_rows = [("모델 출력", corrected_motion)]

    if DEMO_LOWPASS:
        # hips(:3)는 필터에 넣지 않는다 (교정 대상이 아닌 통과값). lowpass_window는 87차원을
        # 받으면 RuntimeError로 즉시 실패하므로 이 슬라이싱이 곧 계약이다.
        q_lp, lp_stats = lowpass.lowpass_window(corrected_motion[:, 3:])
        corrected_motion = torch.cat([corrected_motion[:, :3], q_lp], dim=1)
        stage_rows.append((f"+ 저역통과({lowpass.LP_MODE}, w={lp_stats['window']})", corrected_motion))
    else:
        lp_stats = None

    if DEMO_PROJECTION:
        q_pr, proj_stats = projection.project_window(
            physics, corrected_motion[:, :3], corrected_motion[:, 3:], COLLIDING_PAIRS)
        corrected_motion = torch.cat([corrected_motion[:, :3], q_pr], dim=1)
        stage_rows.append((f"+ 사영(K={projection.PROJ_K}, w={projection.PROJ_OMEGA})", corrected_motion))
    else:
        proj_stats = None

    print(f"\n[파이프라인] 모델 → "
          f"{'저역통과 → ' if DEMO_LOWPASS else ''}{'사영' if DEMO_PROJECTION else ''}".rstrip(' →'))
    print(f"  {'단계':<26}{'최대 침투':>10}{'충돌 프레임':>12}{'지터':>10}")
    print(f"  {'입력(주입됨)':<24}{maxpen_b:>10.2f}{ncoll_b:>10}/{demo_motion.shape[0]:<3}"
          f"{jitter_report(demo_motion):>10.4f}")
    stage_log = []
    for name, m in stage_rows:
        mp, nc = depth_report(m)
        jt = jitter_report(m)
        print(f"  {name:<24}{mp:>10.2f}{nc:>10}/{m.shape[0]:<3}{jt:>10.4f}")
        stage_log.append({"stage": name, "max_pen_cm": round(mp, 3),
                          "collision_frames": nc, "jitter": round(jt, 4)})

    maxpen_a, ncoll_a = depth_report(corrected_motion)

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
        # 'manual'로 뭉뚱그리지 않고 실제 분할 확인 결과를 기록한다 — 나중에 이 데모가
        # 정직했는지(학습 데이터가 아니었는지) meta만 보고 판정할 수 있어야 한다.
        "source_split": ("test (held-out)" if is_heldout
                         else "manual (NOT in held-out test split)"),
        "held_out": is_heldout,
        "checkpoint": ckpt_path,
        "seq_len": SEQ_LEN,
        "demo_scenario": DEMO_SCENARIO,
        "demo_seed": DEMO_SEED,
        "min_depth_cm_used": min_depth,   # clamp 후 실제 채택 기준
        "inject": inject_meta,
        "max_pen_before_cm": round(maxpen_b, 2),
        "max_pen_after_cm": round(maxpen_a, 2),
        "collision_frames_before": ncoll_b,
        "collision_frames_after": ncoll_a,
        # 어떤 후처리 구성으로 만든 결과인지 — 이게 없으면 같은 체크포인트의 서로 다른
        # 파이프라인 결과가 meta만 보고 구분되지 않는다 (CSV의 proj_mode/lp_mode와 같은 역할).
        "pipeline": {
            "order": "model -> lowpass -> projection",
            "lowpass": ({"mode": lowpass.LP_MODE, "window": lp_stats["window"],
                         "edge": lowpass.LP_EDGE,
                         "delta_deg_mean": round(lp_stats["delta_deg_mean"], 4)}
                        if lp_stats is not None else "off"),
            "projection": ({"k": projection.PROJ_K, "omega": projection.PROJ_OMEGA,
                            "margin_cm": projection.PROJ_MARGIN_CM,
                            "pairs": projection.PROJ_PAIRS,
                            "frames_touched": proj_stats["frames_touched"],
                            "residual_max_cm": round(proj_stats["residual_max_cm"], 4),
                            "move_cm": round(proj_stats["move_cm"], 4)}
                           if proj_stats is not None else "off"),
        },
        "stages": stage_log,
    }
    with open(os.path.join(results_dir, "demo_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 5. 결과 애니메이션 저장 (요청 시). 렌더링은 viz_motion의 COMPARE 뷰어를 그대로 쓴다.
    gif_path = save_compare_gif(results_dir) if DEMO_SAVE_GIF else None

    print(f"이 데모 재현: DEMO_SEED={DEMO_SEED} python AI_model/demo_maker.py "
          f"(+ TARGET_FILE='{target_file}' 고정 시 완전 동일)")
    if gif_path:
        print(f"저장된 결과: {os.path.relpath(gif_path)} (gif) + sample_original.pt / "
              f"sample_corrected.pt / demo_meta.json")
    print("준비 완료! 실시간 재생으로 보려면 'python AI_model/viz_motion.py compare' 를 실행하세요 "
          "(지터·뭉개짐은 정지 그림으로 판단할 수 없습니다).")


if __name__ == "__main__":
    create_demo()
