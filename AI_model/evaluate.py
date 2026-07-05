import os
import json
import csv
import argparse
import datetime
import torch
import numpy as np
import itertools
from scipy.spatial.transform import Rotation as R

# 기존 프로젝트 모듈 로드
from dataset_pipeline import (PARENTS, BONE_NAMES, BONE_MAP, BONE_RADII, get_split_files,
                              find_run_dir_by_config, find_latest_checkpoint_in, list_available_runs)
from models import PVTVAE
from physics_module import DifferentiablePhysics
# 평가 대상 실험은 train.py에 설정된 손실 가중치로 선택한다 (mtime이 아니라 설정 기반).
# → train.py에서 LAMBDA_PHYS를 과거 실험 값으로 바꾸면 그 실험을 다시 평가할 수 있다.
from train import LAMBDA_RECON, LAMBDA_PHYS, BETA_KL

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 30

# =====================================================================
# [실행 모드 스위치] VS Code의 "Run Python File" 버튼으로 실행할 때는
# 터미널 인자(--corrupt)를 줄 수 없으므로, 아래 값을 직접 바꿔서 모드를 선택하세요.
#   RUN_CORRUPT_MODE = False -> clean 모드 (held-out 원본 그대로 평가)
#   RUN_CORRUPT_MODE = True  -> corrupt 모드 (인위적 자기충돌 주입 후 충돌 제거 성능 측정)
# (참고: 터미널에서 `python evaluate.py --corrupt` 로 실행하면 이 값과 무관하게 corrupt로 동작)
# =====================================================================
RUN_CORRUPT_MODE = True


def load_run_config(run_dir):
    """train.py가 해당 run 폴더에 남긴 실험용 lambda 설정을 읽는다 (없으면 빈 dict)."""
    p = os.path.join(run_dir, "run_config.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def append_results_csv(row, csv_path="evaluate_results.csv"):
    """
    람다 설정별 '테스트셋 집계' 지표를 CSV 한 줄로 누적 기록한다 (실험 기록용).
    컬럼이 추가/변경되면 기존 파일을 새 헤더로 자동 마이그레이션한다
    (과거 행의 새 컬럼은 빈칸 — 이전 실험 기록은 그대로 보존).
    """
    fields = ["timestamp", "mode", "lambda_recon", "lambda_phys", "epochs", "n_test",
              "collision_before_mean", "collision_after_mean",
              "clean_frames_before_pct", "clean_frames_after_pct",
              "max_pen_before_cm", "max_pen_after_cm",
              "mean_pen_before_cm", "mean_pen_after_cm",
              "depth_removal_pct",
              "mpjpe_cm_mean", "mpjpe_cm_std",
              "mae_deg_mean", "mae_deg_std",
              "jitter_before_mean", "jitter_after_mean",
              "bonelen_after_cm_mean"]
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


def calculate_mae(motion_orig, motion_corr):
    """두 모션의 쿼터니언 회전 간 각도 차이(도). 입력: [F, 87]"""
    F = motion_orig.shape[0]
    q_orig = motion_orig[:, 3:].reshape(F, len(BONE_NAMES), 4)
    q_corr = motion_corr[:, 3:].reshape(F, len(BONE_NAMES), 4)
    dot = torch.clamp(torch.abs(torch.sum(q_orig * q_corr, dim=-1)), max=1.0)
    return (2 * torch.acos(dot) * (180.0 / np.pi)).mean().item()


def calculate_acceleration_jitter(global_pos):
    """FK로 복원한 관절 위치 [F, J, 3]의 가속도 크기 평균 (cm/frame^2)."""
    if global_pos.shape[0] < 3:
        return 0.0
    accel = global_pos[2:] - 2 * global_pos[1:-1] + global_pos[:-2]
    return torch.norm(accel, dim=-1).mean().item() * 100.0


def inject_arm_collision(motion, angle_deg=80.0, bone='LeftUpperArm'):
    """
    (--corrupt 모드) 깨끗한 모션에 인위적 자기충돌(clipping)을 주입한다.
    관절별 위치가 없는 새 양식이므로 상박(LeftUpperArm) 로컬 쿼터니언을 몸통 안쪽으로
    강하게 회전시킨다. FK가 하박/손까지 전파하여 팔이 몸통을 파고들게 만든다.
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


def evaluate(corrupt=False):
    mode = "corrupt" if corrupt else "clean"
    print(f"⏳ [V6] Held-out 테스트셋 전체 모델 성능(집계) 평가 시작 — mode='{mode}'\n")

    # ---- 경로/모델/데이터 준비 --------------------------------------
    motions_dir = "processed_motions_VMC" if os.path.exists("processed_motions_VMC") else "../processed_motions_VMC"
    # train.py에 설정된 손실 가중치와 일치하는 run 폴더를 선택하고, 그 안에서 가장 큰 epoch 가중치를 사용한다.
    run_dir = find_run_dir_by_config(LAMBDA_RECON, LAMBDA_PHYS, BETA_KL)
    if run_dir is None:
        target = f"recon{LAMBDA_RECON:g}_phys{LAMBDA_PHYS:g}_kl{BETA_KL:g}"
        print(f"❌ 설정과 일치하는 학습 폴더(checkpoints/{target}/)를 찾을 수 없습니다.")
        avail = list_available_runs()
        if avail:
            print(f"   사용 가능한 실험 폴더: {avail}")
            print("   → train.py의 LAMBDA_RECON/LAMBDA_PHYS/BETA_KL를 위 이름 중 하나에 맞춰 다시 실행하세요.")
        else:
            print("   아직 학습된 실험이 없습니다. 먼저 train.py를 실행하세요.")
        return
    ckpt_path = find_latest_checkpoint_in(run_dir)

    model = PVTVAE(input_dim=87, output_dim=84, latent_dim=64).to(DEVICE)
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

    cfg = load_run_config(run_dir)
    lam_recon = cfg.get("lambda_recon", "N/A")
    lam_phys = cfg.get("lambda_phys", "N/A")
    run_epochs = cfg.get("epochs", "N/A")   # 100ep/200ep 시대 구분용 (run_config.json에 기록됨)

    # ---- 테스트셋 전체 순회하며 파일별 지표 수집 --------------------
    col_before, col_after = [], []
    mpjpe_list, mae_list = [], []
    jit_before, jit_after = [], []
    bonelen_after = []
    n_used = 0

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

    with torch.no_grad():
        for i, fpath in enumerate(test_files):
            motion = torch.load(fpath)
            if motion.shape[0] < SEQ_LEN:
                continue
            clean = motion[:SEQ_LEN]                        # [30, 87] 깨끗한 정답(ground truth)

            # --corrupt: 모델 입력에 인위적 자기충돌 주입 / 아니면 clean 그대로 입력
            model_input = inject_arm_collision(clean) if corrupt else clean
            recon, _, _ = model(model_input.to(DEVICE).unsqueeze(0))  # [1, 30, 87]
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

            # (2) MPJPE(cm): 교정 결과 vs '깨끗한 정답'과의 위치 오차
            #     clean 모드=원본 보존 오차, corrupt 모드=원래 자세로의 복원 오차
            mpjpe_list.append((torch.norm(gp_clean - gp_corr, dim=-1) * 100.0).mean().item())

            # (3) MAE(deg): 교정 결과 vs 깨끗한 정답의 회전 차이
            mae_list.append(calculate_mae(clean, corr))

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
                print(f"  ... {i + 1}/{len(test_files)} 파일 처리")

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

    # 선형 침투 지표 집계
    clean_before_pct = 100.0 * (1.0 - n_coll_frames_before / max(n_frames_total, 1))
    clean_after_pct = 100.0 * (1.0 - n_coll_frames_after / max(n_frames_total, 1))
    mean_pen_before = depth_sum_before / max(n_coll_frames_before, 1)   # 충돌 프레임 평균 깊이(cm)
    mean_pen_after = depth_sum_after / max(n_coll_frames_after, 1)
    depth_removal_pct = (100.0 * (1.0 - depth_sum_after / depth_sum_before)
                         if depth_sum_before > 0 else None)

    # ---- 결과 출력 --------------------------------------------------
    mpjpe_label = "복원 오차(vs 깨끗한 정답)" if corrupt else "원본 보존 오차"

    print("\n" + "=" * 60)
    print(f"물리 기반 모션 교정(PVTVAE) — Held-out 테스트셋 집계 result [mode='{mode}']")
    print("=" * 60)
    print(f"  [실험 설정] LAMBDA_RECON = {lam_recon} | LAMBDA_PHYS = {lam_phys}")
    print(f"  [평가 규모] held-out 테스트 파일 {n_used}개 (각 앞 {SEQ_LEN}프레임)")
    if corrupt:
        print("  [손상 주입] LeftUpperArm 로컬 회전 +80° → 팔이 몸통을 파고드는 자기충돌 주입")

    print("\n[1] 전신 물리적 무결성 (Physical Plausibility) — 평균")
    print(f"  ▶ 총 {len(ALL_PAIRS)}개 전신 관절 페어")
    print(f"     - Before 충돌(평균, {'손상 입력' if corrupt else 'clean 입력'}) : {cb_m:.6f}")
    print(f"     - After  충돌(평균, 모델 출력) : {ca_m:.6f}")
    if corrupt and cb_m > 0:
        print(f"     - 충돌 제거율(구형 제곱 지표, 과대 평가 경향) : {(1 - ca_m / cb_m) * 100:.1f}%")
    print(f"  ▶ 뼈 길이 변동성(평균) : {bl_m:.4f} cm  (고정 오프셋 → 0 수렴)")

    print("\n[1-b] 해석 가능한 충돌 지표 — 선형 깊이(물리 단위), 테스트셋 전체")
    print(f"  ▶ 충돌 없는 프레임 비율 : Before {clean_before_pct:.1f}% → After {clean_after_pct:.1f}%  (R1: 100%가 목표)")
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

    print("\n" + "-" * 60)
    print("[2] 모션 품질 및 유사도 (Motion Quality) — 평균 ± 표준편차")
    print(f"  ▶ MPJPE({mpjpe_label}) : {mp_m:.2f} ± {mp_s:.2f} cm")
    print(f"  ▶ 회전 각도 오차 (MAE)     : {ma_m:.2f} ± {ma_s:.2f} 도")
    print(f"  ▶ Jitter(참고 지표)        : Before {jb_m:.4f} / After {ja_m:.4f} cm/frame²")
    print("=" * 60)

    # ---- 실험 기록: 테스트셋 집계 지표를 CSV 한 줄로 저장 ----------
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
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
        "jitter_before_mean": round(jb_m, 4),
        "jitter_after_mean": round(ja_m, 4),
        "bonelen_after_cm_mean": round(bl_m, 4),
    }
    csv_path = append_results_csv(row)
    print(f"📝 테스트셋 집계 지표가 '{csv_path}'에 기록되었습니다 "
          f"(mode={mode}, lambda_recon={lam_recon}, lambda_phys={lam_phys}, n_test={n_used}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PVTVAE held-out 테스트셋 평가")
    parser.add_argument("--corrupt", action="store_true",
                        help="held-out 입력에 인위적 자기충돌을 주입해 '충돌 제거 성능'을 측정")
    args = parser.parse_args()
    # 터미널에서 --corrupt를 주면 우선 적용, 아니면 파일 상단 RUN_CORRUPT_MODE 값 사용
    #  → VS Code "Run Python File" 버튼: 상단 RUN_CORRUPT_MODE만 바꾸면 됨
    evaluate(corrupt=(args.corrupt or RUN_CORRUPT_MODE))
