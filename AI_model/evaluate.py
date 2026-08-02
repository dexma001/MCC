import os
import json
import csv
import time
import argparse
import datetime
import random
import torch
import numpy as np
import itertools
from scipy.spatial.transform import Rotation as R

# 기존 프로젝트 모듈 로드
from dataset_pipeline import (PARENTS, BONE_NAMES, BONE_MAP, BONE_RADII, get_split_files,
                              make_run_name, find_run_dir_by_config, find_latest_checkpoint_in,
                              list_available_runs)
# 기본 아키텍처 모델을 '아키텍처 중립' 심볼(MODEL_CLASS)로 로드한다.
# → PVTVAE_baseline/evaluate.py 같은 위성 진입점이 이 심볼 하나만 바꿔치기하면
#   같은 평가 스위트를 다른 아키텍처로 재사용할 수 있다 (평가 방법론은 바이트 단위로 동일).
from models import TransformerDenoiserCompat as MODEL_CLASS
from physics_module import DifferentiablePhysics
import corruption
# 평가 대상 실험은 train.py에 설정된 손실 가중치 + run 태그로 선택한다 (mtime이 아니라 설정 기반).
# → train.py에서 LAMBDA_*/DECLIP_MODE를 과거 실험 값으로 바꾸면 그 실험을 다시 평가할 수 있다.
from train import LAMBDA_RECON, LAMBDA_PHYS, BETA_KL, RUN_TAG, COLLIDING_PAIRS as MONITORED_PAIRS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 30

# =====================================================================
# [평가 시나리오] §4.1 유형별 평가 스위트 — run마다 시나리오별 CSV 행이 1줄씩 기록된다.
#   clean      : held-out 원본 그대로 (항등 보존 / 클린 입력에 충돌을 새로 만들지 않는지)
#   legacy80   : 구형 고정 주입 (LeftUpperArm 로컬 Z +80°, 전 프레임) — 학습 분포에서
#                의도적으로 제외된 held-out 심층 손상. 시대 간 비교 연속성 + 오염 검사용.
#                이 시나리오에서 MPJPE(vs 클린)는 참고 수치일 뿐이다 (깊은 지속형에서는
#                복원이 아니라 관통 제거 + 의도 보존이 정직한 잣대 — 설계 문서 참조).
#   transient  : corruption.py의 sin-ramp 일시적 주입 (고정 평가 시드, 재현 가능)
#   persistent : corruption.py의 깊이 목표(1~4cm) 지속 주입 (고정 평가 시드)
#
# VS Code "Run Python File" 버튼 사용자: 아래 리스트를 직접 수정하면 됩니다.
# 터미널: python evaluate.py [--corrupt | --scenarios clean,transient] [--limit N]
# =====================================================================
RUN_SCENARIOS = ["clean", "legacy80", "transient", "persistent"]

# 평가 주입 파라미터 추첨용 고정 시드 — 파일 인덱스별로 파생되어 실행마다 동일한 손상이 재현됨.
# (학습의 CORRUPTION_SEED와 다른 값이므로 학습이 본 손상 '인스턴스'는 평가에 재등장하지 않음.
#  transient/persistent 시나리오는 '같은 분포'에 대한 복구 성능을, legacy80은 분포-외 심층
#  손상을 측정한다는 역할 구분이 설계에 명시되어 있다.)
EVAL_SEED = 20260707
EVAL_CORRUPTION_CFG = corruption.make_cfg()   # 주입 '형태' 파라미터는 설계 기본값 고정

# =====================================================================
# [항목 2] 3DPCK 임계값(cm) — 3D HPE 표준 지표(관절 정확도 통과율)를 우리 스케일에 맞춘 것.
#   필드 표준 150mm는 우리 스케일에서 무의미하다(MPJPE가 이미 4~6cm라 전부 100%로 포화).
#   1/2/5cm는 max_pen과 '같은 축(cm)'이라 침투 깊이와 나란히 읽힌다. 발명 상수 0개.
#
# 🚨 시나리오 게이팅 (설계상 필수): PCK는 '클린 정답'을 기준으로 하므로 거리에 단조인 지표다.
#   따라서 persistent/legacy80(지속형 심층 손상)에서는 "의도적 포즈를 원래대로 되돌릴수록
#   점수가 오르는" 역상관이 발생한다 — 제품 목표는 복원이 아니라 '최소 사영'이기 때문이다.
#   이는 지금 헤드라인에서 강등하는 persistent MPJPE와 정확히 같은 함정이므로,
#   PCK는 아래 두 시나리오에서만 계산하고 나머지는 CSV에 빈칸으로 남긴다.
# =====================================================================
PCK_THRESHOLDS_CM = (1.0, 2.0, 5.0)
PCK_SCENARIOS = ("clean", "transient")

# [항목 1] 추론 시간 측정에서 버릴 워밍업 파일 수 (CUDA 컨텍스트 생성/커널 캐싱 구간 제외)
TIMING_WARMUP_FILES = 5


def load_run_config(run_dir):
    """train.py가 해당 run 폴더에 남긴 실험용 lambda/주입 설정을 읽는다 (없으면 빈 dict)."""
    p = os.path.join(run_dir, "run_config.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def append_results_csv(row, csv_path="evaluate_results.csv"):
    """
    시나리오×람다 설정별 '테스트셋 집계' 지표를 CSV 한 줄로 누적 기록한다 (실험 기록용).
    컬럼이 추가/변경되면 기존 파일을 새 헤더로 자동 마이그레이션한다
    (과거 행의 새 컬럼은 빈칸 — 이전 실험 기록은 그대로 보존).
    """
    fields = ["timestamp", "mode", "run_tag", "lambda_recon", "lambda_phys", "epochs", "n_test",
              "collision_before_mean", "collision_after_mean",
              "clean_frames_before_pct", "clean_frames_after_pct",
              "max_pen_before_cm", "max_pen_after_cm",
              "mean_pen_before_cm", "mean_pen_after_cm",
              "depth_removal_pct",
              "mpjpe_cm_mean", "mpjpe_cm_std",
              "mae_deg_mean", "mae_deg_std",
              "intent_mae_deg",
              "jitter_before_mean", "jitter_after_mean",
              "bonelen_after_cm_mean",
              # ---- Tier 1 신규 컬럼 (2026-07-28) ----------------------------------
              # 🚨 스키마 규칙: 신규 컬럼은 '반드시 맨 뒤에만' 추가한다. 절대 중간 삽입 금지.
              #    evaluate_visualize.xlsx가 CSV의 '컬럼 위치'를 직접 참조해 만들어졌기 때문에,
              #    중간에 끼워 넣으면 기존 추출 범위가 전부 어긋난다. 맨 뒤 추가면 기존 1~24열
              #    인덱스가 보존되어 기존 시각화가 무수정으로 동작한다. (Tier 2 이후에도 동일 규칙)
              # [항목 4] 학습이 실제 최적화하는 4쌍 한정 헤드라인 충돌 지표
              "max_pen4_before_cm", "max_pen4_after_cm",
              "clean_frames4_before_pct", "clean_frames4_after_pct",
              "depth_removal4_pct",
              # [항목 2] 3DPCK — clean/transient 시나리오에서만 기록(그 외 빈칸, 아래 게이팅 참조)
              "pck_1cm", "pck_2cm", "pck_5cm",
              "pck_frame_1cm", "pck_frame_2cm", "pck_frame_5cm",
              # [항목 3] 동역학 보존 (전 시나리오 기록 — clean에서는 do-no-harm 수치가 된다)
              "intent_dyn_cm",
              # [항목 1] 추론 시간 (R4) — ⚠️ 배포 지연시간이 아니라 '상한 프록시'
              "infer_ms_window_mean", "infer_ms_window_p95", "infer_ms_per_frame", "infer_device"]
    exists = os.path.exists(csv_path)
    if exists:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames or [])
            old_rows = list(reader)
        if old_fields != fields:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in old_rows:
                    w.writerow({k: r.get(k, "") for k in fields})
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)
    return csv_path


def get_all_eval_pairs():
    """전신 무결성 검사용 페어 자동 생성 (위상 거리 2 초과)"""
    adj = {b: [] for b in BONE_NAMES}
    for c, p in PARENTS.items():
        if p:
            adj[p].append(c)
            adj[c].append(p)

    def get_dist(start, target):
        if start == target:
            return 0
        q = [(start, 0)]
        visited = {start}
        while q:
            curr, d = q.pop(0)
            if curr == target:
                return d
            for nxt in adj[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, d + 1))
        return 999

    pairs = []
    capsules = [(PARENTS[b], b) for b in BONE_NAMES if PARENTS[b] is not None]
    for (p1, c1), (p2, c2) in itertools.combinations(capsules, 2):
        min_dist = min(get_dist(p1, p2), get_dist(p1, c2), get_dist(c1, p2), get_dist(c1, c2))
        if min_dist > 2:
            pairs.append(((p1, c1), (p2, c2)))
    return pairs


def per_joint_angle_deg(motion_a, motion_b):
    """두 모션의 관절별 쿼터니언 각도 차이(도). 입력: [F, 87] → 반환 [F, J]."""
    F = motion_a.shape[0]
    qa = motion_a[:, 3:].reshape(F, len(BONE_NAMES), 4)
    qb = motion_b[:, 3:].reshape(F, len(BONE_NAMES), 4)
    dot = torch.clamp(torch.abs(torch.sum(qa * qb, dim=-1)), max=1.0)
    return 2 * torch.acos(dot) * (180.0 / np.pi)


def calculate_mae(motion_orig, motion_corr):
    """두 모션의 쿼터니언 회전 간 각도 차이(도) 전체 평균. 입력: [F, 87]"""
    return per_joint_angle_deg(motion_orig, motion_corr).mean().item()


def calculate_intent_mae(motion_out, motion_in, injected_bone_idx):
    """
    [부수 변화(collateral) 지표] 주입되지 '않은' 관절들에 대해 (모델 출력 vs 손상 입력)의
    회전 차이(도). 낮을수록 좋다 — 모델이 손상 부위만 고치고 나머지는 건드리지 않았다는 뜻.
    기존 지표(클린 대비 근접도)만으로는 "팔을 홱 잡아떼고도 좋은 점수"인 과잉 보정을
    구조적으로 볼 수 없어서 추가된 지표 (R2).

    ⚠️ 이름 주의 (2026-07-28 재정의): 이것은 '의도(intent)' 지표가 아니라 '부수 변화' 지표다.
       측정하는 것은 "주입되지 않은 관절이 얼마나 덩달아 움직였는가"이며, 의도의 의미론이
       아니다. 한계: (1) 정당한 협응 교정(어깨와 함께 움직여야 하는 경우)도 벌점 처리한다,
       (2) injected_bone_idx 라벨은 '우리가 주입했기 때문에' 아는 값이라 실제 배포
       스트림에서는 계산할 수 없다, (3) 동역학/타이밍 정보가 없다.
       (2)(3)을 보완하는 것이 calculate_intent_dyn 이다.

    📌 CSV 컬럼명은 'intent_mae_deg' 그대로 유지한다 — append_results_csv의 마이그레이션이
       키 기반 복사(r.get(k, ""))라서 컬럼명을 바꾸면 기존 행들의 값이 조용히 소실된다.
       따라서 개명은 '표시 라벨과 문서'에서만 수행한다.
    """
    ang = per_joint_angle_deg(motion_out, motion_in)      # [F, J]
    mask = torch.ones(len(BONE_NAMES), dtype=torch.bool)
    mask[injected_bone_idx] = False
    return ang[:, mask].mean().item()


def calculate_acceleration_jitter(global_pos):
    """FK로 복원한 관절 위치 [F, J, 3]의 가속도 크기 평균 (cm/frame^2)."""
    if global_pos.shape[0] < 3:
        return 0.0
    accel = global_pos[2:] - 2 * global_pos[1:-1] + global_pos[:-2]
    return torch.norm(accel, dim=-1).mean().item() * 100.0


def calculate_intent_dyn(gp_out, gp_in):
    """
    [동역학 보존 지표, R2] 출력과 '손상 입력'의 가속도 벡터 차이 평균 (cm/frame²).

        intent_dyn = mean ‖Δ²(gp_out) − Δ²(gp_in)‖ × 100

    위 calculate_acceleration_jitter와 연산자([1,-2,1] 시간축 2차 차분)는 완전히 같지만,
    '한 신호의 절대 평활도'가 아니라 '두 신호의 가속도 벡터 차이'라는 점이 핵심이다.
    참조가 바뀌면서 평활도 prior → 보존 prior 로 의미가 뒤집힌다.

    왜 이 형태인가:
      - Δ²는 상수·1차 성분을 소거한다 → 정당한 de-clip이 수행하는 '상수 재배치'(팔을 몸에서
        일정 거리 띄우는 교정)에 불변. 올바른 교정을 의도 훼손으로 벌하지 않는다.
      - 지속형(persistent) 주입은 그 자체가 상수 오프셋이라 가속도 ≈ 0 → 주입 관절이 자동
        상쇄된다. 즉 이 시나리오에서는 bone_idx 마스크 없이 계산 가능하며, 라벨이 존재하지
        않는 '실제 배포 스트림'에서도 그대로 계산할 수 있다 (intent_mae의 구조적 약점 우회).
      - 크기 차(‖Δ²(out)‖−‖Δ²(in)‖)가 아니라 벡터 차인 이유: 크기만 보면 방향이 달라져도
        0이 될 수 있다. 벡터 차여야 타이밍/위상 변화를 잡는다.

    ⚠️ 한계 (단독 사용 금지): 정적 의미 오류에 맹목이다. 정지 제스처가 다른 정지 제스처로
       뒤바뀌어도 양쪽 모두 Δ²=0이라 이 지표는 0을 보고한다. 반드시 정적 포즈/의미 항과
       병용해야 한다 (짝이 될 cOKS는 Tier 2). 또한 transient 주입은 그 자체에 가속도 서명이
       있으므로, 그 시나리오에서 이 값은 '글리치 제거분'과 '퍼포먼스 왜곡분'이 섞여 있다.
    """
    if gp_out.shape[0] < 3:
        return 0.0
    acc_out = gp_out[2:] - 2 * gp_out[1:-1] + gp_out[:-2]
    acc_in = gp_in[2:] - 2 * gp_in[1:-1] + gp_in[:-2]
    return torch.norm(acc_out - acc_in, dim=-1).mean().item() * 100.0


def inject_arm_collision(motion, angle_deg=80.0, bone='LeftUpperArm'):
    """
    (legacy80 시나리오) 깨끗한 모션에 인위적 자기충돌(clipping)을 주입한다.
    상박(LeftUpperArm) 로컬 쿼터니언을 몸통 안쪽으로 강하게 회전시키고 FK가 하박/손까지
    전파하여 팔이 몸통을 파고들게 만든다. §4.1 학습 주입과의 오염 방지를 위해 이 조합
    (고정 본/축/80°/전 프레임)은 학습 분포에서 의도적으로 제외되어 있다 — 수정 금지.
    입력/출력: [F, 87]  (원본은 보존, 손상된 복사본을 반환)
    """
    idx = BONE_MAP[bone]
    sl = slice(3 + idx * 4, 3 + idx * 4 + 4)
    out = motion.clone()
    delta = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * np.radians(angle_deg))  # 로컬 Z축 스윙
    q = out[:, sl].numpy()                                # [F, 4]
    q_new = (R.from_quat(q) * delta).as_quat()            # 로컬 프레임에서 추가 회전
    out[:, sl] = torch.tensor(q_new, dtype=out.dtype)
    return out


def make_scenario_input(scenario, clean, file_idx, physics_engine):
    """
    시나리오별 모델 입력 생성. 반환: (model_input [30,87], meta dict|None)
    meta에는 주입 본 인덱스 등이 담겨 의도 보존 지표 계산에 쓰인다.
    """
    if scenario == "clean":
        return clean, None
    if scenario == "legacy80":
        return inject_arm_collision(clean), dict(type='legacy80',
                                                 bone_idx=BONE_MAP['LeftUpperArm'])
    if scenario == "transient":
        rng = random.Random(EVAL_SEED + 10007 * file_idx + 1)
        return corruption.inject_transient(clean, physics_engine, EVAL_CORRUPTION_CFG,
                                           rng, MONITORED_PAIRS)
    if scenario == "persistent":
        rng = random.Random(EVAL_SEED + 10007 * file_idx + 2)
        return corruption.inject_persistent(clean, physics_engine, EVAL_CORRUPTION_CFG,
                                            rng, MONITORED_PAIRS)
    raise ValueError(f"알 수 없는 시나리오: {scenario}")


def evaluate(scenarios=None, limit=0):
    scenarios = scenarios or RUN_SCENARIOS
    print(f"⏳ [V7] Held-out 테스트셋 유형별 시나리오 평가 시작 — scenarios={scenarios}\n")

    # ---- 경로/모델/데이터 준비 --------------------------------------
    motions_dir = "processed_motions_VMC" if os.path.exists("processed_motions_VMC") else "../processed_motions_VMC"
    # train.py에 설정된 손실 가중치/태그와 일치하는 run 폴더에서 가장 큰 epoch 가중치를 사용한다.
    run_dir = find_run_dir_by_config(LAMBDA_RECON, LAMBDA_PHYS, BETA_KL, tag=RUN_TAG)
    if run_dir is None:
        target = make_run_name(LAMBDA_RECON, LAMBDA_PHYS, BETA_KL, tag=RUN_TAG)
        print(f"❌ 설정과 일치하는 학습 폴더(checkpoints/{target}/)를 찾을 수 없습니다.")
        avail = list_available_runs()
        if avail:
            print(f"   사용 가능한 실험 폴더: {avail}")
            print("   → train.py의 LAMBDA_*/DECLIP_MODE를 위 이름 중 하나에 맞춰 다시 실행하세요.")
        else:
            print("   아직 학습된 실험이 없습니다. 먼저 train.py를 실행하세요.")
        return
    ckpt_path = find_latest_checkpoint_in(run_dir)

    model = MODEL_CLASS(input_dim=87, output_dim=84, latent_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()  # 결정론적 추론(mu 사용)
    print(f"✅ 모델 로드: {os.path.relpath(ckpt_path)}")

    physics_engine = DifferentiablePhysics(PARENTS, BONE_RADII)
    ALL_PAIRS = get_all_eval_pairs()

    # 🚨 학습에 쓰지 않은 held-out 파일만 평가 대상으로 사용
    test_files = [f for f in get_split_files(motions_dir, split='test')]
    if not test_files:
        print("❌ held-out(test) 파일이 없습니다.")
        return
    if limit:
        test_files = test_files[:limit]
        print(f"⚠️ --limit {limit}: 테스트 파일 {len(test_files)}개만 사용 (스모크/디버그 모드)")

    cfg = load_run_config(run_dir)
    lam_recon = cfg.get("lambda_recon", "N/A")
    lam_phys = cfg.get("lambda_phys", "N/A")
    run_epochs = cfg.get("epochs", "N/A")   # 100ep/200ep 시대 구분용 (run_config.json에 기록됨)
    run_tag = cfg.get("run_tag", RUN_TAG)

    for scenario in scenarios:
        _eval_one_scenario(scenario, model, physics_engine, ALL_PAIRS, test_files,
                           lam_recon, lam_phys, run_epochs, run_tag)


def _eval_one_scenario(scenario, model, physics_engine, ALL_PAIRS, test_files,
                       lam_recon, lam_phys, run_epochs, run_tag):
    corrupt = scenario != "clean"

    # ---- 테스트셋 전체 순회하며 파일별 지표 수집 --------------------
    col_before, col_after = [], []
    mpjpe_list, mae_list, intent_list = [], [], []
    intent_dyn_list = []            # [항목 3] 동역학 보존 (전 시나리오)
    jit_before, jit_after = [], []
    bonelen_after = []
    n_used = 0
    n_inject_fallback = 0   # persistent 시나리오에서 목표 깊이 실패로 클린 폴백된 파일 수

    # 해석 가능한 선형 침투 지표 누적기 (물리 단위 cm — 제곱/스케일 없음)
    n_frames_total = 0
    n_coll_frames_before = 0        # 침투가 1개 페어라도 존재하는 프레임 수
    n_coll_frames_after = 0
    depth_sum_before = 0.0          # 프레임별 최대 침투 깊이(cm)의 총합 (깊이 기준 제거율용)
    depth_sum_after = 0.0
    max_pen_before = 0.0            # 테스트셋 전체 최악 침투 깊이(cm)
    max_pen_after = 0.0
    pair_max_before = torch.zeros(len(ALL_PAIRS))   # 페어별 최악 침투 (교정 전/후)
    pair_max_after = torch.zeros(len(ALL_PAIRS))

    # [항목 4] 학습이 '실제로' 최적화하는 4쌍(MONITORED_PAIRS = train.COLLIDING_PAIRS) 한정 누적기.
    #   역할 분리: 4쌍 = 학습 충실도(de-clip 성능) / 112쌍 = 전신 do-no-harm(부작용 감시).
    #   지금까지 한 숫자가 두 역할을 겸해 생긴 해석 혼동을 지표 분리로 해소한다 (§8-A).
    #   ⚠️ 기존 112쌍 지표는 그대로 병기한다 — 치환하면 과거 44행과 비교 가능성이 파괴된다.
    n_coll4_before = 0
    n_coll4_after = 0
    depth4_sum_before = 0.0
    depth4_sum_after = 0.0
    max_pen4_before = 0.0
    max_pen4_after = 0.0

    # [항목 2] 3DPCK 누적기 — 파일별 평균의 평균이 아니라 '전역 카운터'로 센다
    #   (clean_frames_pct와 동일한 이벤트 의미론 유지). pck_gate가 False면 계산 자체를 건너뛴다.
    pck_gate = scenario in PCK_SCENARIOS

    # [항목 1] 추론 시간 측정 (R4). 모델 forward '만' 잰다 — FK/충돌 계산은 평가 전용이라 제외.
    #   앞 TIMING_WARMUP_FILES개는 CUDA 컨텍스트 초기화/커널 캐싱 때문에 통계에서 버린다.
    win_ms = []
    n_joint_total = 0                                    # 총 (프레임 × 관절) 수
    pck_joint_hit = {t: 0 for t in PCK_THRESHOLDS_CM}    # 관절 단위 통과 수
    pck_frame_hit = {t: 0 for t in PCK_THRESHOLDS_CM}    # 프레임 '전원 통과' 수 (R1의 이벤트 형태)
    joint_err_sum = torch.zeros(len(BONE_NAMES))         # 관절별 오차 합 (콘솔 top-3 최악 관절용)

    with torch.no_grad():
        for i, fpath in enumerate(test_files):
            motion = torch.load(fpath)
            if motion.shape[0] < SEQ_LEN:
                continue
            clean = motion[:SEQ_LEN]                        # [30, 87] 깨끗한 정답(ground truth)

            # 시나리오별 모델 입력 생성 (주입 meta 포함)
            model_input, meta = make_scenario_input(scenario, clean, i, physics_engine)
            if meta is not None and meta.get('fallback'):
                n_inject_fallback += 1
            # [항목 1] forward 구간만 계측. CUDA는 비동기이므로 synchronize로 커널 완료를 기다린다
            #   (없으면 '큐에 넣는 시간'만 재게 되어 값이 비현실적으로 작아진다).
            model_in_dev = model_input.to(DEVICE).unsqueeze(0)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            _t0 = time.perf_counter()
            recon, _, _ = model(model_in_dev)                          # [1, 30, 87]
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            win_ms.append((time.perf_counter() - _t0) * 1000.0)
            corr = recon.squeeze(0).cpu()                   # [30, 87] 모델 교정 결과

            # FK로 관절 월드 좌표 복원 [30, 21, 3]
            gp_clean = physics_engine.compute_global_pos_tensor(clean[:, :3], clean[:, 3:])
            gp_input = physics_engine.compute_global_pos_tensor(model_input[:, :3], model_input[:, 3:])
            gp_corr = physics_engine.compute_global_pos_tensor(corr[:, :3], corr[:, 3:])

            # (1) 충돌: Before = 모델 입력(손상 or clean), After = 모델 출력
            dict_in = {b: gp_input[:, BONE_MAP[b]] for b in BONE_NAMES}
            dict_c = {b: gp_corr[:, BONE_MAP[b]] for b in BONE_NAMES}
            col_before.append(physics_engine.get_collision_loss(dict_in, ALL_PAIRS).item())
            col_after.append(physics_engine.get_collision_loss(dict_c, ALL_PAIRS).item())

            # (1-b) 해석 가능한 선형 침투 지표: 페어별 깊이(cm) → 프레임별 최대 깊이
            dep_in = physics_engine.get_penetration_depths(dict_in, ALL_PAIRS) * 100.0   # [30, P] cm
            dep_c = physics_engine.get_penetration_depths(dict_c, ALL_PAIRS) * 100.0
            fmax_in = dep_in.max(dim=1).values   # [30] 프레임별 최대 침투 깊이
            fmax_c = dep_c.max(dim=1).values

            n_frames_total += fmax_in.shape[0]
            n_coll_frames_before += int((fmax_in > 1e-4).sum())
            n_coll_frames_after += int((fmax_c > 1e-4).sum())
            depth_sum_before += float(fmax_in.sum())
            depth_sum_after += float(fmax_c.sum())
            max_pen_before = max(max_pen_before, float(fmax_in.max()))
            max_pen_after = max(max_pen_after, float(fmax_c.max()))
            pair_max_before = torch.maximum(pair_max_before, dep_in.max(dim=0).values)
            pair_max_after = torch.maximum(pair_max_after, dep_c.max(dim=0).values)

            # (1-c) [항목 4] 동일한 집계를 '학습 대상 4쌍'에만 적용 (헤드라인 de-clip 지표).
            #   ⚠️ 4쌍은 112쌍의 부분집합이 아니다 (2026-07-28 실측 확인):
            #      get_all_eval_pairs()는 캡슐을 (PARENTS[b], b)로만 만드는데 PARENTS['Chest']='Spine'
            #      이므로, 손으로 정의한 긴 몸통 캡슐 (Hips→Chest)은 112쌍에 등장할 수 없다.
            #      즉 4쌍 중 2쌍(몸통↔좌/우 하박)은 112쌍이 '구조적으로 볼 수 없는' 충돌이다.
            #      → max_pen4 > max_pen(112) 이 정상적으로 발생할 수 있다. 부분집합 가정 금지.
            dep4_in = physics_engine.get_penetration_depths(dict_in, MONITORED_PAIRS) * 100.0
            dep4_c = physics_engine.get_penetration_depths(dict_c, MONITORED_PAIRS) * 100.0
            f4max_in = dep4_in.max(dim=1).values   # [30] 프레임별 최대 침투 깊이 (4쌍 한정)
            f4max_c = dep4_c.max(dim=1).values

            n_coll4_before += int((f4max_in > 1e-4).sum())
            n_coll4_after += int((f4max_c > 1e-4).sum())
            depth4_sum_before += float(f4max_in.sum())
            depth4_sum_after += float(f4max_c.sum())
            max_pen4_before = max(max_pen4_before, float(f4max_in.max()))
            max_pen4_after = max(max_pen4_after, float(f4max_c.max()))

            # (2) MPJPE(cm): 교정 결과 vs '깨끗한 정답'과의 위치 오차
            #     clean=원본 보존 오차 / transient=복원 오차 / persistent·legacy80=참고 수치
            #     관절별 거리 [30, 21]을 변수로 빼서 아래 3DPCK가 그대로 재사용한다 (재계산 없음).
            joint_dist_cm = torch.norm(gp_clean - gp_corr, dim=-1) * 100.0   # [30, 21] cm
            mpjpe_list.append(joint_dist_cm.mean().item())

            # (2-b) [항목 2] 3DPCK: 같은 거리 텐서에 임계값 비교만 추가 → 평균이 이벤트 비율이 된다.
            #   MPJPE의 1/21 희석(한 관절 20cm 오차 ≈ 전신 1cm 오차) 문제를 분포 형태로 보완.
            if pck_gate:
                n_joint_total += joint_dist_cm.numel()
                joint_err_sum += joint_dist_cm.sum(dim=0)
                for t in PCK_THRESHOLDS_CM:
                    ok = joint_dist_cm < t                      # [30, 21]
                    pck_joint_hit[t] += int(ok.sum())           # 관절 단위
                    pck_frame_hit[t] += int(ok.all(dim=1).sum())  # 프레임 전원 통과

            # (3) MAE(deg): 교정 결과 vs 깨끗한 정답의 회전 차이
            mae_list.append(calculate_mae(clean, corr))

            # (3-b) 의도 보존: 주입되지 않은 관절들의 (출력 vs 손상 입력) 회전 차이
            if meta is not None and meta.get('type') != 'clean':
                intent_list.append(calculate_intent_mae(corr, model_input, meta['bone_idx']))

            # (3-c) [항목 3] 동역학 보존: 출력과 손상 입력의 가속도 '벡터 차'
            #   clean 시나리오에서는 gp_input == gp_clean 이므로 "모델이 멀쩡한 동역학을
            #   얼마나 왜곡하는가"를 재는 do-no-harm 수치가 된다 → 전 시나리오 기록.
            intent_dyn_list.append(calculate_intent_dyn(gp_corr, gp_input))

            # (4) Jitter(참고 지표): 부드러움
            jit_before.append(calculate_acceleration_jitter(gp_input))
            jit_after.append(calculate_acceleration_jitter(gp_corr))

            # (5) 뼈 길이 변동성(고정 오프셋이므로 구조적으로 0)
            bl = []
            for c, p in PARENTS.items():
                if p:
                    ci, pi = BONE_MAP[c], BONE_MAP[p]
                    bl.append(torch.std(torch.norm(gp_corr[:, ci] - gp_corr[:, pi], dim=-1)).item() * 100.0)
            bonelen_after.append(float(np.mean(bl)))

            n_used += 1
            if (i + 1) % 50 == 0:
                print(f"  [{scenario}] ... {i + 1}/{len(test_files)} 파일 처리")

    def ms(a):
        a = np.array(a, dtype=np.float64)
        return float(a.mean()), float(a.std())

    cb_m, _ = ms(col_before)
    ca_m, _ = ms(col_after)
    mp_m, mp_s = ms(mpjpe_list)
    ma_m, ma_s = ms(mae_list)
    jb_m, _ = ms(jit_before)
    ja_m, _ = ms(jit_after)
    bl_m, _ = ms(bonelen_after)
    intent_m = float(np.mean(intent_list)) if intent_list else None
    intent_dyn_m = float(np.mean(intent_dyn_list)) if intent_dyn_list else None

    # [항목 1] 추론 시간 집계 — 워밍업 구간 제외 후 평균/p95.
    #   p95를 함께 두는 이유: R4의 실패는 평균이 아니라 '스파이크'에서 발생한다
    #   (한 프레임이라도 예산을 넘기면 드롭 → 시청자에게 보이는 결함).
    timed = win_ms[TIMING_WARMUP_FILES:] if len(win_ms) > TIMING_WARMUP_FILES else win_ms
    if timed:
        infer_ms_mean = float(np.mean(timed))
        infer_ms_p95 = float(np.percentile(timed, 95))
        infer_ms_frame = infer_ms_mean / SEQ_LEN
    else:
        infer_ms_mean = infer_ms_p95 = infer_ms_frame = None

    # 선형 침투 지표 집계
    clean_before_pct = 100.0 * (1.0 - n_coll_frames_before / max(n_frames_total, 1))
    clean_after_pct = 100.0 * (1.0 - n_coll_frames_after / max(n_frames_total, 1))
    mean_pen_before = depth_sum_before / max(n_coll_frames_before, 1)   # 충돌 프레임 평균 깊이(cm)
    mean_pen_after = depth_sum_after / max(n_coll_frames_after, 1)
    depth_removal_pct = (100.0 * (1.0 - depth_sum_after / depth_sum_before)
                         if depth_sum_before > 0 else None)

    # [항목 4] 4쌍 한정 집계 (112쌍과 완전히 같은 공식 — 페어 범위만 다르다)
    clean4_before_pct = 100.0 * (1.0 - n_coll4_before / max(n_frames_total, 1))
    clean4_after_pct = 100.0 * (1.0 - n_coll4_after / max(n_frames_total, 1))
    depth_removal4_pct = (100.0 * (1.0 - depth4_sum_after / depth4_sum_before)
                          if depth4_sum_before > 0 else None)

    # [항목 2] 3DPCK 집계 (게이팅된 시나리오에서만 값이 생기고, 그 외에는 None → CSV 빈칸)
    if pck_gate and n_joint_total > 0:
        n_frames_pck = n_joint_total // len(BONE_NAMES)
        pck_joint = {t: 100.0 * pck_joint_hit[t] / n_joint_total for t in PCK_THRESHOLDS_CM}
        pck_frame = {t: 100.0 * pck_frame_hit[t] / n_frames_pck for t in PCK_THRESHOLDS_CM}
        joint_err_mean = joint_err_sum / max(n_frames_pck, 1)   # 관절별 평균 오차(cm)
    else:
        pck_joint, pck_frame, joint_err_mean = None, None, None

    # ---- 결과 출력 --------------------------------------------------
    scenario_desc = {
        "clean": "클린 입력 (항등 보존 검사)",
        "legacy80": "고정 심층 주입: LeftUpperArm 로컬 Z +80°, 전 프레임 (held-out/오염 검사)",
        "transient": "일시적 sin-ramp 주입 (θ~U[15°,70°], 5~20프레임, 고정 평가 시드)",
        "persistent": "지속적 깊이-목표 주입 (1~4cm, 전-윈도우/반열림 혼합, 고정 평가 시드)",
    }
    mpjpe_label = {"clean": "원본 보존 오차",
                   "transient": "복원 오차(vs 깨끗한 정답)"}.get(
                   scenario, "vs 클린 정답 (지속형에서는 참고 수치)")

    print("\n" + "=" * 60)
    print(f"Held-out 테스트셋 집계 — scenario='{scenario}'  (arch={MODEL_CLASS.__name__})")
    print("=" * 60)
    print(f"  [실험 설정] tag={run_tag} | LAMBDA_RECON = {lam_recon} | LAMBDA_PHYS = {lam_phys}")
    print(f"  [평가 규모] held-out 테스트 파일 {n_used}개 (각 앞 {SEQ_LEN}프레임)")
    print(f"  [시나리오 ] {scenario_desc.get(scenario, scenario)}")
    if scenario == "persistent" and n_inject_fallback:
        print(f"  [주의] 깊이 목표 실패로 클린 폴백된 파일: {n_inject_fallback}개")

    # ── [1] 헤드라인 = 학습 대상 4쌍 (§8-A). de-clip '성능'은 이 블록으로 판단한다. ──
    print(f"\n[1] de-clip 성능 (헤드라인) — 학습 대상 {len(MONITORED_PAIRS)}개 페어, 선형 깊이(cm)")
    print("    ↳ 역할: 학습 충실도. 물리 손실이 실제로 최적화한 바로 그 페어 집합.")
    print(f"  ▶ 충돌 없는 프레임 비율 : Before {clean4_before_pct:.1f}% → After {clean4_after_pct:.1f}%  (R1: 100%가 목표)")
    print(f"  ▶ 최대 침투 깊이(최악)  : Before {max_pen4_before:.2f} cm → After {max_pen4_after:.2f} cm")
    if depth_removal4_pct is not None:
        print(f"  ▶ 깊이 기준 충돌 제거율 : {depth_removal4_pct:.1f}%  (선형)")

    # ── [1-b] 전신 112쌍 = do-no-harm 감시 (디버그). 강등되었을 뿐 값은 불변. ──
    print(f"\n[1-b] 전신 do-no-harm (디버그) — 자동 생성 {len(ALL_PAIRS)}개 페어, 선형 깊이(cm)")
    print("    ↳ 역할: 부작용 감시. 학습이 보지 않는 페어까지 포함하므로 de-clip 성능 지표가 아니다.")
    print(f"       (4쌍은 112쌍의 부분집합이 아님 — 몸통 캡슐 Hips→Chest는 112쌍에 존재하지 않는다)")
    print(f"  ▶ 충돌 없는 프레임 비율 : Before {clean_before_pct:.1f}% → After {clean_after_pct:.1f}%")
    print(f"  ▶ 최대 침투 깊이(최악)  : Before {max_pen_before:.2f} cm → After {max_pen_after:.2f} cm")
    print(f"  ▶ 충돌 프레임 평균 깊이 : Before {mean_pen_before:.2f} cm → After {mean_pen_after:.2f} cm")
    if depth_removal_pct is not None:
        print(f"  ▶ 깊이 기준 충돌 제거율 : {depth_removal_pct:.1f}%  (선형 — 제곱 지표보다 보수적/정직)")
    # 교정 후에도 남은 최악 페어 top-3 (문제 부위 파악용)
    k = min(3, len(ALL_PAIRS))
    top_after = torch.topk(pair_max_after, k=k)
    for depth, idx in zip(top_after.values.tolist(), top_after.indices.tolist()):
        if depth <= 1e-4:
            break
        (p1, c1), (p2, c2) = ALL_PAIRS[idx]
        print(f"     - 교정 후 최악 페어: ({p1}→{c1}) ↔ ({p2}→{c2})  최대 {depth:.2f} cm")
    print(f"  ▶ 구형 제곱 충돌 지표(과대 평가 경향, 시대 간 연속성용) : "
          f"Before {cb_m:.6f} → After {ca_m:.6f}")
    if corrupt and cb_m > 0:
        print(f"     - 구형 제거율 : {(1 - ca_m / cb_m) * 100:.1f}%")
    print(f"  ▶ 뼈 길이 변동성(평균) : {bl_m:.4f} cm  (고정 오프셋 → 0 수렴)")

    # ── [2] 클린 기준 충실도. 지속형에서는 이 블록이 헤드라인이 아니다(아래 [3]으로 강등). ──
    clean_ref_valid = scenario in PCK_SCENARIOS
    print("\n" + "-" * 60)
    if clean_ref_valid:
        print("[2] 모션 품질 및 유사도 (Motion Quality) — 평균 ± 표준편차")
        print(f"  ▶ MPJPE({mpjpe_label}) : {mp_m:.2f} ± {mp_s:.2f} cm")
        print(f"  ▶ 회전 각도 오차 (MAE)     : {ma_m:.2f} ± {ma_s:.2f} 도")
        pj = " / ".join(f"{t:g}cm {pck_joint[t]:.1f}%" for t in PCK_THRESHOLDS_CM)
        pf = " / ".join(f"{t:g}cm {pck_frame[t]:.1f}%" for t in PCK_THRESHOLDS_CM)
        print(f"  ▶ 3DPCK (관절 단위)        : {pj}")
        print(f"  ▶ 3DPCK (프레임 전원 통과) : {pf}   ← R1의 이벤트 형태")
        k3 = min(3, len(BONE_NAMES))
        worst = torch.topk(joint_err_mean, k=k3)
        detail = ", ".join(f"{BONE_NAMES[i]} {v:.2f}cm"
                           for v, i in zip(worst.values.tolist(), worst.indices.tolist()))
        print(f"     - 최악 관절 top-{k3}: {detail}  (FK 사슬 누적 → 말단이 지배적)")
    else:
        # persistent/legacy80: 클린 기준 지표는 '복원할수록 점수가 오르는' 역상관이므로
        # 헤드라인에서 내리고 참고 수치로만 남긴다. CSV 값은 히스토리 보존을 위해 그대로 기록.
        print("[2] 클린 기준 지표 — ⚠️ 참고 수치 (헤드라인 아님)")
        print("    ↳ 지속형 손상에서 '클린 복원'은 제품 목표가 아니다(목표=최소 사영). 이 값이")
        print("      낮을수록 좋다고 해석하면 의도된 포즈를 되돌리는 모델을 우대하게 된다.")
        print(f"  ▶ MPJPE({mpjpe_label}) : {mp_m:.2f} ± {mp_s:.2f} cm")
        print(f"  ▶ 회전 각도 오차 (MAE)     : {ma_m:.2f} ± {ma_s:.2f} 도")
        print("  ▶ 3DPCK                    : 미측정(의도적) — 위와 같은 역상관 사유")

    # ── [3] 의도/퍼포먼스 보존 (R2). 지속형에서는 [1]과 함께 이 블록이 헤드라인이다. ──
    print("\n[3] 의도·퍼포먼스 보존 (R2)" + ("" if clean_ref_valid else "  ← 지속형의 헤드라인"))
    if intent_m is not None:
        print(f"  ▶ 부수 변화(collateral, 비주입 관절: 출력 vs 손상 입력) : {intent_m:.2f} 도  (낮을수록 좋음)")
        print("    ↳ 주의: '의도'가 아니라 '주입되지 않은 관절이 얼마나 덩달아 움직였는가'다.")
        print("       배포 시에는 주입 라벨이 없어 계산 불가 (CSV 컬럼명은 히스토리 보존상 intent_mae_deg 유지).")
    if intent_dyn_m is not None:
        role = "do-no-harm(동역학 왜곡)" if scenario == "clean" else "동역학 보존"
        print(f"  ▶ 동역학 보존 intent_dyn : {intent_dyn_m:.4f} cm/frame²  ({role}, 낮을수록 좋음)")
        print("    ↳ ‖Δ²(출력) − Δ²(입력)‖ — 상수 재배치에 불변이라 정당한 교정을 벌하지 않는다.")
        print("       persistent에서는 마스크 없이 계산되어 '배포 스트림에서도 측정 가능'하다.")
        print("       ⚠️ 정적 의미 오류에 맹목 → 단독 판단 금지, 포즈/의미 항(cOKS, Tier 2)과 병용할 것.")
    print(f"  ▶ Jitter(참고 지표)        : Before {jb_m:.4f} / After {ja_m:.4f} cm/frame²")

    # ── [4] 추론 속도 (R4). 지금까지 한 번도 측정되지 않던 요구사항. ──
    if infer_ms_mean is not None:
        print(f"\n[4] 추론 속도 (R4) — device={DEVICE}, 워밍업 {TIMING_WARMUP_FILES}개 제외 "
              f"(n={len(timed)})")
        print(f"  ▶ 30프레임 윈도우 1회 : 평균 {infer_ms_mean:.2f} ms / p95 {infer_ms_p95:.2f} ms")
        print(f"  ▶ 프레임당 환산       : {infer_ms_frame:.3f} ms/frame")
        print("    ↳ ⚠️ 이 값은 '배포 지연시간'이 아니라 상한 프록시(하한 비용)다. 아직 causal 모드,")
        print("       출력 스무딩 필터, 60↔30fps 리샘플 어댑터가 구현되지 않았고, 30프레임 윈도우를")
        print("       한 번에 처리하는 구조라 실서비스에는 ~1초의 look-ahead 지연이 별도로 존재한다.")
    print("=" * 60)

    # ---- 실험 기록: 시나리오별 집계 지표를 CSV 한 줄로 저장 ----------
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": scenario,
        "run_tag": run_tag or "",
        "lambda_recon": lam_recon,
        "lambda_phys": lam_phys,
        "epochs": run_epochs,
        "n_test": n_used,
        "collision_before_mean": round(cb_m, 6),
        "collision_after_mean": round(ca_m, 6),
        "clean_frames_before_pct": round(clean_before_pct, 2),
        "clean_frames_after_pct": round(clean_after_pct, 2),
        "max_pen_before_cm": round(max_pen_before, 2),
        "max_pen_after_cm": round(max_pen_after, 2),
        "mean_pen_before_cm": round(mean_pen_before, 2),
        "mean_pen_after_cm": round(mean_pen_after, 2),
        "depth_removal_pct": round(depth_removal_pct, 1) if depth_removal_pct is not None else "",
        "mpjpe_cm_mean": round(mp_m, 2),
        "mpjpe_cm_std": round(mp_s, 2),
        "mae_deg_mean": round(ma_m, 2),
        "mae_deg_std": round(ma_s, 2),
        "intent_mae_deg": round(intent_m, 2) if intent_m is not None else "",
        "jitter_before_mean": round(jb_m, 4),
        "jitter_after_mean": round(ja_m, 4),
        "bonelen_after_cm_mean": round(bl_m, 4),
        # ---- Tier 1 신규 컬럼 (순서 고정: 반드시 기존 24개 뒤에만) --------------
        # [항목 4] 학습 대상 4쌍 한정 헤드라인 지표 (112쌍 컬럼은 위에 그대로 병기)
        "max_pen4_before_cm": round(max_pen4_before, 2),
        "max_pen4_after_cm": round(max_pen4_after, 2),
        "clean_frames4_before_pct": round(clean4_before_pct, 2),
        "clean_frames4_after_pct": round(clean4_after_pct, 2),
        "depth_removal4_pct": round(depth_removal4_pct, 1) if depth_removal4_pct is not None else "",
        # [항목 2] 3DPCK — 게이팅되지 않은 시나리오(persistent/legacy80)는 의도적으로 빈칸
        "pck_1cm": round(pck_joint[1.0], 2) if pck_joint else "",
        "pck_2cm": round(pck_joint[2.0], 2) if pck_joint else "",
        "pck_5cm": round(pck_joint[5.0], 2) if pck_joint else "",
        "pck_frame_1cm": round(pck_frame[1.0], 2) if pck_frame else "",
        "pck_frame_2cm": round(pck_frame[2.0], 2) if pck_frame else "",
        "pck_frame_5cm": round(pck_frame[5.0], 2) if pck_frame else "",
        # [항목 3] 동역학 보존 (전 시나리오)
        "intent_dyn_cm": round(intent_dyn_m, 4) if intent_dyn_m is not None else "",
        # [항목 1] 추론 시간 (R4). ⚠️ 상한 프록시 — causal/필터/리샘플 미포함, look-ahead 별도.
        "infer_ms_window_mean": round(infer_ms_mean, 3) if infer_ms_mean is not None else "",
        "infer_ms_window_p95": round(infer_ms_p95, 3) if infer_ms_p95 is not None else "",
        "infer_ms_per_frame": round(infer_ms_frame, 4) if infer_ms_frame is not None else "",
        "infer_device": DEVICE,
    }
    csv_path = append_results_csv(row)
    print(f"📝 시나리오 '{scenario}' 집계 지표가 '{csv_path}'에 기록되었습니다 "
          f"(tag={run_tag}, lambda_recon={lam_recon}, lambda_phys={lam_phys}, n_test={n_used}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="기본 아키텍처 held-out 테스트셋 유형별 시나리오 평가")
    parser.add_argument("--corrupt", action="store_true",
                        help="(구형 호환) legacy80 시나리오만 실행")
    parser.add_argument("--scenarios", type=str, default="",
                        help="쉼표 구분 시나리오 목록 (예: clean,transient). 생략 시 전체 스위트 실행")
    parser.add_argument("--limit", type=int, default=0,
                        help="테스트 파일 수 제한 (0=전체). 스모크/디버그용")
    args = parser.parse_args()

    if args.scenarios:
        chosen = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    elif args.corrupt:
        chosen = ["legacy80"]
    else:
        chosen = RUN_SCENARIOS   # VS Code "Run Python File" 버튼: 파일 상단 리스트 수정
    evaluate(scenarios=chosen, limit=args.limit)
