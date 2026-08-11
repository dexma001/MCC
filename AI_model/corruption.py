"""
§4.1 지도학습 디클리핑용 손상(클리핑) 주입 모듈.

train.py(학습 입력 손상) / evaluate.py(유형별 평가 시나리오)가 이 모듈 하나를 공유하여
주입 구현이 어긋나지 않게 한다. 설계 근거: declip_supervision_problem.md,
collision_after_root_cause_report.md §7 (2026-07-07 합의).

주입 유형 (v1):
  - transient(일시적, 글리치 의미론): sin-ramp 회전. θ(t) = θ_peak·sin(π·(t-s)/T),
    θ_peak ~ U[15°, 70°] (평가용 80°는 오염 방지를 위해 학습 분포에서 제외),
    지속 길이 T ~ U[5, 20] 프레임, 시작 프레임은 윈도우 경계 클리핑 허용
    (진입/이탈만 보이는 부분 관측 케이스가 자연 생성됨 — 의도된 부산물).
  - persistent(지속적, 의도/체형 불일치 의미론): 상수 오프셋.
    각도가 아닌 '관통 깊이'를 기준으로 목표 1~4cm를 탐색한다 — 같은 각도라도 포즈에 따라
    충돌량이 달라(실증됨) 각도 고정으로는 손상 강도가 통제되지 않기 때문. 각도는 통제
    변수가 아니므로 사다리(ladder) + 이분 보간으로 깊이 범위에 들어가는 각도를 찾는다.
    ⚠️ 깊이 판정은 '주입 구간의 (coverage, 중앙값)'으로 한다 — 윈도우 전체 max 하나로
    판정하면 한 프레임만 깊게 뚫린 후보가 채택되어 나머지가 무관통으로 남는다
    (2026-08-07 수정 전 실측: 주입 구간 프레임의 62%가 침투 0 → 수정 후 0.0%).
    구간은 전-윈도우(다수) + 반열림(onset/offset, 소수) 혼합 — 완전 포함 구간은
    윈도우 스케일에서 일시적 시그니처가 되므로 사용하지 않는다 (보고서 §7-B-2).

타겟은 두 유형 모두 '클린 원본'(v1 Option A): 지속형은 작은 깊이로 제한되어
클린 타겟 ≈ 최소 사영이므로 정답지가 정직하다.

[구현 노트 — 속도] FK 비용은 텐서 크기가 아니라 '호출 횟수'(21개 본 파이썬 루프)가
지배한다. 따라서 후보(샘플 × 각도 × 축)를 전부 쌓아 라운드당 '배치 FK 1회'로 깊이를
평가한다. 샘플별 순차 rejection 대비 배치당 FK 호출이 ~100회 → ~5회로 줄어든다.
"""
import math

import torch

from dataset_pipeline import BONE_MAP

# 주입 대상 본: 팔 4개(팔↔몸통 / 팔↔팔 페어 유발) + 상부 다리 2개(다리↔다리 페어 유발)
DEFAULT_INJECT_BONES = [
    'LeftUpperArm', 'RightUpperArm', 'LeftLowerArm', 'RightLowerArm',
    'LeftUpperLeg', 'RightUpperLeg',
]


def make_cfg(clean_ratio=0.5,
             transient_ratio=0.3,
             theta_range_deg=(15.0, 70.0),
             transient_dur_range=(5, 20),
             transient_min_depth_cm=0.3,
             transient_max_tries=6,
             persistent_depth_range_cm=(1.0, 4.0),
             persistent_angle_ladder_deg=(8.0, 15.0, 25.0, 38.0, 55.0, 78.0),
             persistent_halfopen_ratio=0.35,
             halfopen_min_len=15,
             persistent_max_rounds=6,
             persistent_min_coverage=1.0,
             persistent_draws_per_round=6,
             inject_bones=None):
    """
    손상 주입 설정 dict 생성 (JSON 직렬화 가능 → run_config.json에 그대로 기록).
      - clean_ratio        : 전체 샘플 중 손상 없이 통과시키는 비율 (항등 보존 학습, R2 앵커)
      - transient_ratio    : '손상 샘플' 중 일시적 비율 (합의: 0.3 → 30/70)
      - transient_min_depth_cm : 일시적 주입이 최소 이만큼은 관통하도록 재추첨하는 하한
                                 (2026-07-06 실증: 고정 회전은 포즈에 따라 충돌 0이 될 수 있음)
      - persistent_angle_ladder_deg : 깊이 목표 탐색용 각도 후보 사다리 (상한 78° —
                                      평가용 80° 근방을 학습 분포에서 배제)
      - persistent_halfopen_ratio : 지속 주입 중 반열림(onset/offset) 구간 비율.
                                    나머지는 전-윈도우(무문맥 최소사영 학습의 핵심 샘플).
      - halfopen_min_len   : 반열림 구간의 최소 길이(프레임). 윈도우 절반 이상을 관통 상태로
                             유지해 '윈도우 안에서 손상이 완결되지 않음'이라는 지속형 본질을 보존.
      - persistent_min_coverage : 지속 주입 구간에서 '실제로 관통한 프레임' 최소 비율.
                             1.0이면 구간 전 프레임이 관통해야 채택. 구판에는 이 조건이
                             없어(윈도우 max만 검사) 구간 프레임의 62%가 무관통이었다
                             (실측 2026-08-07). 지속형의 정의가 '구간 내내 관통'이므로
                             이것이 유형을 유형답게 만드는 핵심 파라미터다.
      - persistent_draws_per_round : 라운드당 추첨하는 (본, 축) 조합 수. coverage 조건이
                             붙으면 임의의 본/축으로는 구간 내내 관통시키기 어려워
                             1개만 뽑으면 클린 폴백이 급증한다(실측 38~49%). FK는
                             라운드당 1회 배치이므로 이 값을 올려도 FK 호출 수는 늘지
                             않는다(후보 텐서만 커진다). 6에서 폴백률이 구판 수준으로 복귀.
    """
    return dict(
        clean_ratio=clean_ratio,
        transient_ratio=transient_ratio,
        theta_range_deg=list(theta_range_deg),
        transient_dur_range=list(transient_dur_range),
        transient_min_depth_cm=transient_min_depth_cm,
        transient_max_tries=transient_max_tries,
        persistent_depth_range_cm=list(persistent_depth_range_cm),
        persistent_angle_ladder_deg=list(persistent_angle_ladder_deg),
        persistent_halfopen_ratio=persistent_halfopen_ratio,
        halfopen_min_len=halfopen_min_len,
        persistent_max_rounds=persistent_max_rounds,
        persistent_min_coverage=persistent_min_coverage,
        persistent_draws_per_round=persistent_draws_per_round,
        inject_bones=list(inject_bones or DEFAULT_INJECT_BONES),
    )


# ------------------------------------------------------------------
# 쿼터니언 유틸 (physics_module과 동일한 [x, y, z, w] / Unity 규약)
# ------------------------------------------------------------------
def _quat_mul(q1, q2):
    """[..., 4] ⊗ [..., 4]. q1 ⊗ q2 = 'q2를 먼저, q1을 나중에' (로컬 추가 회전은 q ⊗ delta)."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)


def _rand_unit_axis(rng):
    """등방(isotropic) 랜덤 단위 회전축. 평가의 고정 로컬-Z 주입과 달리 축도 랜덤화한다."""
    while True:
        v = [rng.gauss(0.0, 1.0) for _ in range(3)]
        n = math.sqrt(sum(x * x for x in v))
        if n > 1e-6:
            return [x / n for x in v]


def _apply_local_delta(window, bone_idx, frames, axis, angles_rad):
    """
    window [S, 87]의 특정 본에, 지정 프레임들만 로컬 추가 회전(축·각)을 적용한 복사본 반환.
    inject_arm_collision과 같은 규약: q_new = q ⊗ delta (로컬 프레임에서 추가 회전).
    """
    out = window.clone()
    sl = slice(3 + bone_idx * 4, 3 + bone_idx * 4 + 4)
    dev, dt = window.device, window.dtype

    ang = torch.tensor(angles_rad, dtype=dt, device=dev)          # [n]
    ax = torch.tensor(axis, dtype=dt, device=dev)                 # [3]
    half = ang * 0.5
    deltas = torch.cat([ax.unsqueeze(0) * torch.sin(half).unsqueeze(-1),
                        torch.cos(half).unsqueeze(-1)], dim=-1)   # [n, 4]

    idx = torch.tensor(frames, dtype=torch.long, device=dev)
    q = out[idx, sl]                                              # [n, 4]
    qn = _quat_mul(q, deltas)
    out[idx, sl] = qn / (qn.norm(dim=-1, keepdim=True) + 1e-8)
    return out


def _apply_local_delta_multi(window, bone_idx, frames, axis_angles_list):
    """
    window [S, 87] '하나'에 대해 (축, 각도열) 후보 K개를 배치 연산 한 번으로 적용 → [K, S, 87].
    후보별 _apply_local_delta 반복 호출과 원소별 연산이 동일해 결과가 비트 단위로 같다
    (2026-07-13 A/B 검증). 프로파일 결과 후보 생성이 corrupt_batch 시간의 ~47%였고
    (배치당 ~370회 × 0.17ms), 그 비용이 텐서 크기가 아니라 호출 횟수에 비례했기 때문에
    K개 후보(지속형 사다리 12개 등)를 한 번에 만드는 것이 핵심 절감이다.
    """
    K = len(axis_angles_list)
    sl = slice(3 + bone_idx * 4, 3 + bone_idx * 4 + 4)
    dev, dt = window.device, window.dtype

    out = window.unsqueeze(0).repeat(K, 1, 1)                     # [K, S, 87]
    ang = torch.tensor([aa[1] for aa in axis_angles_list], dtype=dt, device=dev)  # [K, n]
    ax = torch.tensor([aa[0] for aa in axis_angles_list], dtype=dt, device=dev)   # [K, 3]
    half = ang * 0.5
    deltas = torch.cat([ax.unsqueeze(1) * torch.sin(half).unsqueeze(-1),
                        torch.cos(half).unsqueeze(-1)], dim=-1)   # [K, n, 4]

    idx = torch.tensor(frames, dtype=torch.long, device=dev)
    q = out[:, idx, sl]                                           # [K, n, 4]
    qn = _quat_mul(q, deltas)
    out[:, idx, sl] = qn / (qn.norm(dim=-1, keepdim=True) + 1e-8)
    return out


def _frame_depths_batched(windows, physics, pairs):
    """
    후보 윈도우 묶음 [M, S, 87]의 '프레임별' 최대 선형 관통 깊이(cm) [M, S] 반환.
    라운드당 FK 1회로 모든 후보를 평가하는 것이 이 모듈의 속도 핵심.
    """
    with torch.no_grad():
        dep = physics.get_penetration_depths_from_quats(
            windows[..., :3], windows[..., 3:], pairs) * 100.0    # [M, S, n_pairs]
    return dep.amax(dim=-1)                                       # [M, S]


def _depths_batched(windows, physics, pairs):
    """
    후보 윈도우 묶음 [M, S, 87]의 윈도우별 최대 선형 관통 깊이(cm) [M] 반환.
    (주입되지 않은 프레임은 클린 ≈ 무관통이므로 전 프레임 max ≈ 주입 구간 max.)

    ⚠️ 이 '윈도우 하나당 스칼라 하나' 요약은 transient(sin-ramp: 애초에 한 봉우리만
    깊게 뚫는 것이 정의)에만 적합하다. persistent는 구간 '내내' 관통해야 하므로
    max 하나로는 품질을 판정할 수 없다 — _seg_stats를 쓴다. (2026-08-07)
    """
    return _frame_depths_batched(windows, physics, pairs).amax(dim=-1)   # [M]


def _seg_stats(frame_depths, f0, f1):
    """
    프레임별 깊이 [S] 에서 주입 구간 [f0, f1] 의 (coverage, 중앙값, 최대값)을 낸다.
      - coverage : 구간 프레임 중 실제로 관통(>0)한 비율. '지속성'의 척도.
      - 중앙값   : 전형적인 프레임의 침투 깊이. 목표 범위(1~4cm) 판정에 쓴다.
    윈도우 max 대신 이 둘을 보는 이유: 같은 각도라도 포즈가 프레임마다 달라 깊이가
    출렁이므로, max만 보면 '한 프레임만 3cm 뚫린' 후보가 채택되어 나머지 프레임이
    무관통으로 남는다 (실측 2026-08-07: 주입 구간 프레임의 62%가 침투 0이었다).
    """
    seg = frame_depths[f0:f1 + 1]
    cov = float((seg > 1e-4).float().mean())
    return cov, float(seg.median()), float(seg.max())


# ------------------------------------------------------------------
# 일시적(transient) 주입 — sin-ramp (그룹 단위 배치 처리)
# ------------------------------------------------------------------
def _inject_transient_group(windows, physics, cfg, rng, pairs):
    """
    windows: [S,87] 텐서 리스트 → [(손상 복사본, meta), ...] (입력 순서 유지).
    라운드마다 샘플별로 (타이밍, θ_peak, 본, 축±) 후보 2개를 만들고 배치 FK 1회로 깊이를
    평가한다. 최소 관통(transient_min_depth_cm) 미달이면 재추첨; 전부 실패하면 마지막
    추첨을 그대로 사용한다 (무충돌 섭동도 유효한 복원 샘플 — meta['collided']=False).
    """
    n = len(windows)
    S = windows[0].shape[0]
    results = [None] * n
    last = [None] * n
    pending = list(range(n))

    for round_i in range(1, cfg['transient_max_tries'] + 1):
        if not pending:
            break
        cands, owners, cmetas = [], [], []
        for j in pending:
            dur = rng.randint(cfg['transient_dur_range'][0], cfg['transient_dur_range'][1])
            start = rng.randint(-(dur // 2), S - 1 - dur // 2)   # 경계 클리핑 허용
            theta_peak = rng.uniform(cfg['theta_range_deg'][0], cfg['theta_range_deg'][1])
            bone = rng.choice(cfg['inject_bones'])
            axis = _rand_unit_axis(rng)

            f0, f1 = max(0, start), min(S - 1, start + dur)
            frames = list(range(f0, f1 + 1))
            angles = [math.radians(theta_peak) * math.sin(math.pi * (f - start) / dur)
                      for f in frames]

            # 축 방향이 몸 바깥쪽일 수 있으므로 ±axis 두 후보를 같은 라운드에 평가
            # (θ 분포 자체는 설계값 U[15°,70°] 유지; 방향만 보정)
            ax_list = (axis, [-a for a in axis])
            multi = _apply_local_delta_multi(windows[j], BONE_MAP[bone], frames,
                                             [(ax, angles) for ax in ax_list])
            for k, ax in enumerate(ax_list):
                cands.append(multi[k])
                owners.append(j)
                cmetas.append(dict(type='transient', bone=bone, bone_idx=BONE_MAP[bone],
                                   frames=(f0, f1), theta_peak_deg=round(theta_peak, 2),
                                   axis=[round(a, 4) for a in ax], tries=round_i))

        depths = _depths_batched(torch.stack(cands), physics, pairs)

        accepted = set()
        for c_i in range(len(cands)):
            j = owners[c_i]
            if j in accepted or results[j] is not None:
                continue
            meta = dict(cmetas[c_i], max_depth_cm=round(float(depths[c_i]), 3), collided=True)
            last[j] = (cands[c_i], meta)
            if depths[c_i] >= cfg['transient_min_depth_cm']:
                results[j] = (cands[c_i], meta)
                accepted.add(j)
        pending = [j for j in pending if j not in accepted]

    for j in pending:   # 모든 시도 무충돌 → 마지막 추첨 사용
        cand, meta = last[j]
        meta['collided'] = False
        results[j] = (cand, meta)
    return results


def inject_transient(window, physics, cfg, rng, pairs):
    """단일 윈도우 [S,87] 진입점 (evaluate.py 등에서 사용). 원본은 보존."""
    return _inject_transient_group([window], physics, cfg, rng, pairs)[0]


# ------------------------------------------------------------------
# 지속적(persistent) 주입 — 깊이 목표(1~4cm) 각도 탐색 (그룹 단위 배치 처리)
# ------------------------------------------------------------------
def _inject_persistent_group(windows, physics, cfg, rng, pairs):
    """
    windows: [S,87] 텐서 리스트 → [(손상 복사본, meta), ...] (입력 순서 유지).
    샘플별로 (본, ±축)을 추첨하고 각도 사다리 전체를 후보로 만들어 배치 FK 1회로 깊이를
    평가한다. 사다리에 목표 범위(1~4cm)가 없으면 범위를 감싸는 두 각도 사이를 이분 보간
    (다음 라운드), 양방향 모두 무침투면 본/축 재추첨. persistent_max_rounds 안에 실패하면
    클린 폴백(meta['fallback']=True) — 호출 측이 카운트해 비율 왜곡을 감시.
    """
    n = len(windows)
    S = windows[0].shape[0]
    lo, hi = cfg['persistent_depth_range_cm']
    min_cov = cfg.get('persistent_min_coverage', 0.9)
    ladder = list(cfg['persistent_angle_ladder_deg'])
    results = [None] * n

    # 구간(전-윈도우/반열림)은 샘플당 1회 확정 — 탐색 재추첨과 무관하게 유지
    seg_info = []
    for _ in range(n):
        if rng.random() < cfg['persistent_halfopen_ratio']:
            L = rng.randint(cfg['halfopen_min_len'], S - 1)
            if rng.random() < 0.5:
                seg_info.append((S - L, S - 1, 'onset'))    # 클린 → 진입 후 윈도우 끝까지
            else:
                seg_info.append((0, L - 1, 'offset'))       # 시작부터 관통 → 이탈
        else:
            seg_info.append((0, S - 1, 'full'))

    # pending 상태: refine=None이면 새 (본,축) 추첨 + 사다리, refine=(bone, axis, [각도들])이면 보간 후보만
    pending = [dict(j=j, refine=None, draws=0) for j in range(n)]

    for round_i in range(1, cfg['persistent_max_rounds'] + 1):
        if not pending:
            break
        cands, owners, cmetas = [], [], []
        for p in pending:
            j = p['j']
            f0, f1, seg = seg_info[j]
            frames = list(range(f0, f1 + 1))
            if p['refine'] is not None:
                draws = [p['refine']]                      # (bone, [축], 각도열)
            else:
                # 라운드마다 (본, 축)을 여러 개 뽑는다. coverage 조건을 붙인 뒤로는
                # '아무 본/축'이나 구간 내내 관통시키지 못하기 때문 — 1개만 뽑으면
                # 대부분 실패해 클린 폴백으로 떨어진다 (실측: 폴백 45%).
                # FK는 어차피 라운드당 1회 배치이므로 후보를 늘려도 호출 수는 그대로다.
                draws = []
                for _ in range(cfg.get('persistent_draws_per_round', 3)):
                    ax0 = _rand_unit_axis(rng)
                    draws.append((rng.choice(cfg['inject_bones']),
                                  [ax0, [-a for a in ax0]], ladder))
                p['draws'] += 1
            for bone, axes, angle_list in draws:
                axes = [axes] if isinstance(axes[0], float) else list(axes)
                combos = [(ax, theta) for ax in axes for theta in angle_list]
                multi = _apply_local_delta_multi(
                    windows[j], BONE_MAP[bone], frames,
                    [(ax, [math.radians(theta)] * len(frames)) for ax, theta in combos])
                for k, (ax, theta) in enumerate(combos):
                    cands.append(multi[k])
                    owners.append(id(p))
                    cmetas.append(dict(bone=bone, axis=ax, theta=theta,
                                       frames=(f0, f1), seg=seg))

        frame_depths = _frame_depths_batched(torch.stack(cands), physics, pairs)  # [M, S]

        # 샘플별 후보 결과 취합. 깊이 판정은 '윈도우 max'가 아니라 주입 구간의
        # (coverage, 중앙값)으로 한다 — 지속형의 정의가 '구간 내내 관통'이기 때문.
        by_p = {id(p): [] for p in pending}
        for c_i in range(len(cands)):
            m = cmetas[c_i]
            f0, f1 = m['frames']
            cov, med, mx = _seg_stats(frame_depths[c_i], f0, f1)
            by_p[owners[c_i]].append((c_i, m, cov, med, mx))

        still = []
        for p in pending:
            j = p['j']
            entries = by_p[id(p)]
            # 1) coverage를 만족하면서 깊이 중앙값이 목표 범위인 후보 → 중앙에 가장 가까운 것
            ok = [e for e in entries if e[2] >= min_cov and lo <= e[3] <= hi]
            if ok:
                mid = (lo + hi) * 0.5
                c_i, m, cov, med, mx = min(ok, key=lambda e: abs(e[3] - mid))
                results[j] = (cands[c_i], dict(type='persistent', bone=m['bone'],
                                               bone_idx=BONE_MAP[m['bone']],
                                               frames=m['frames'], seg=m['seg'],
                                               theta_deg=round(m['theta'], 2),
                                               axis=[round(a, 4) for a in m['axis']],
                                               max_depth_cm=round(mx, 3),
                                               median_depth_cm=round(med, 3),
                                               coverage=round(cov, 3),
                                               tries=round_i, collided=True))
                continue
            # 2) 같은 축에서 목표 범위를 감싸는 (얕음, 과침투) 쌍이 있으면 그 사이를 보간해 재시도.
            #    보간의 조종 대상도 중앙값 — max로 조종하면 1)의 채택 조건과 어긋난다.
            refine = None
            for ax_key in set(tuple(m['axis']) for (_c, m, _v, _md, _mx) in entries):
                ax_entries = sorted([(m['theta'], md) for (_c, m, _v, md, _mx) in entries
                                     if tuple(m['axis']) == ax_key])
                below = [(t, d) for t, d in ax_entries if d < lo]
                above = [(t, d) for t, d in ax_entries if d > hi]
                if above:
                    t_lo = max((t for t, _d in below), default=0.0)   # 얕은 쪽 (없으면 0°)
                    t_hi = min(t for t, _d in above)                  # 과침투 쪽
                    mids = [t_lo + (t_hi - t_lo) * f for f in (0.25, 0.5, 0.75)]
                    refine = (entries[0][1]['bone'], list(ax_key), mids)
                    break
            if refine is not None:
                p['refine'] = refine
                still.append(p)
                continue
            # 3) 모든 각도·양방향이 무침투/얕음 → 이 본/축으로는 불가, 재추첨
            p['refine'] = None
            still.append(p)
            # 차선책 보관: coverage를 만족하는 후보 중 목표 범위에 가장 가까운 것.
            #   라운드를 모두 소진했을 때 '클린 폴백'으로 버리는 대신 이것을 쓴다.
            #   (구판은 폴백 시 손상 샘플이 통째로 클린이 되어 지속형 비율이 조용히 줄었다)
            cov_ok = [e for e in entries if e[2] >= min_cov and e[3] > 0.0]
            if cov_ok:
                mid = (lo + hi) * 0.5
                cand = min(cov_ok, key=lambda e: abs(e[3] - mid))
                prev = p.get('best')
                if prev is None or abs(cand[3] - mid) < abs(prev[3] - mid):
                    p['best'] = cand
                    p['best_round'] = round_i
                    # cands는 라운드마다 새로 만들어지므로 텐서를 지금 붙잡아 둔다
                    # (인덱스만 저장하면 다음 라운드에 다른 후보를 가리키게 된다).
                    p['best_window'] = cands[cand[0]]
        pending = still

    for p in pending:   # 라운드 소진
        j = p['j']
        f0, f1, seg = seg_info[j]
        best = p.get('best')
        if best is not None:
            # 목표 범위(1~4cm)에는 못 들었지만 coverage는 만족하는 차선 후보를 쓴다.
            # 지속형의 본질(구간 내내 관통)은 지켜지므로 클린으로 버리는 것보다 정직하다.
            c_i, m, cov, med, mx = best
            results[j] = (p['best_window'], dict(
                type='persistent', bone=m['bone'], bone_idx=BONE_MAP[m['bone']],
                frames=m['frames'], seg=m['seg'], theta_deg=round(m['theta'], 2),
                axis=[round(a, 4) for a in m['axis']],
                max_depth_cm=round(mx, 3), median_depth_cm=round(med, 3),
                coverage=round(cov, 3), tries=p.get('best_round', round_i),
                collided=True, out_of_range=True))
        else:
            # coverage를 만족하는 후보가 아예 없었다 → 클린 폴백 (항등 보존 샘플로 학습)
            results[j] = (windows[j].clone(), dict(type='clean', fallback=True, seg=seg))
    return results


def inject_persistent(window, physics, cfg, rng, pairs):
    """단일 윈도우 [S,87] 진입점 (evaluate.py 등에서 사용). 원본은 보존."""
    return _inject_persistent_group([window], physics, cfg, rng, pairs)[0]


# ------------------------------------------------------------------
# 배치 단위 진입점 (train.py가 매 배치 호출)
# ------------------------------------------------------------------
def corrupt_batch(clean_batch, physics, cfg, rng, pairs):
    """
    clean_batch [B, S, 87] → (모델 입력용 손상 배치, 샘플별 meta 리스트).
    타겟은 항상 원본 clean_batch (호출 측이 그대로 보관).
    혼합비: clean_ratio는 전체 대비, transient_ratio는 '손상 샘플' 중 비율.
      (기본 0.5 / 0.3 → 전체 분포: 클린 50% / 일시적 15% / 지속적 35%)
    """
    B = clean_batch.shape[0]
    out = clean_batch.clone()
    metas = [dict(type='clean') for _ in range(B)]

    tr_idx, pe_idx = [], []
    for i in range(B):
        if rng.random() < cfg['clean_ratio']:
            continue
        (tr_idx if rng.random() < cfg['transient_ratio'] else pe_idx).append(i)

    if tr_idx:
        for i, (w, m) in zip(tr_idx, _inject_transient_group(
                [clean_batch[i] for i in tr_idx], physics, cfg, rng, pairs)):
            out[i] = w
            metas[i] = m
    if pe_idx:
        for i, (w, m) in zip(pe_idx, _inject_persistent_group(
                [clean_batch[i] for i in pe_idx], physics, cfg, rng, pairs)):
            out[i] = w
            metas[i] = m
    return out, metas


# ------------------------------------------------------------------
# 학습 루프용 백그라운드 프리페치 (파이프라인 병렬화)
# ------------------------------------------------------------------
def corrupted_batches(dataloader, physics, cfg, rng, pairs, prefetch=2):
    """
    DataLoader 배치를 백그라운드 스레드에서 '순차적으로' 손상시켜
    (clean_batch, corrupted_batch, metas)를 내놓는 제너레이터.

    목적: 주입은 CPU 작업(배치당 ~수십 ms)이라 메인 루프에서 직접 호출하면 그 시간 동안
    GPU가 통째로 논다. 이 제너레이터는 GPU가 배치 N을 학습하는 동안 배치 N+1의 주입을
    준비해 주입 비용을 GPU 시간 뒤로 숨긴다 (에폭 시간 ≈ max(주입, GPU) + ε).

    재현성 보장: 데이터 순회와 rng 소비가 '단일 스레드에서 순차'로 일어나므로,
    메인 루프에서 corrupt_batch를 직접 부르던 기존 방식과 배치 순서·난수 소비 순서·
    주입 결과가 완전히 동일하다 (연산 자체는 그대로, 실행 '시점'만 겹칠 뿐).
    (메인 스레드는 numpy/torch CPU 전역 난수를 쓰지 않아야 한다 — 현 train.py 충족.)
    """
    import queue
    import threading

    q = queue.Queue(maxsize=prefetch)
    _END = object()
    errors = []

    def _producer():
        try:
            for batch in dataloader:
                corrupted, metas = corrupt_batch(batch, physics, cfg, rng, pairs)
                q.put((batch, corrupted, metas))
        except BaseException as e:   # 예외는 소비 측에서 다시 던진다
            errors.append(e)
        finally:
            q.put(_END)

    threading.Thread(target=_producer, daemon=True).start()
    while True:
        item = q.get()
        if item is _END:
            if errors:
                raise errors[0]
            return
        yield item
