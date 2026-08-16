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

[구현 노트 — 속도 (2026-08-11 개편)]
  주입은 학습 시간을 지배하는 비용이다(개편 전 실측 210~309ms/batch vs GPU 학습 스텝 37.6ms).
  비용의 정체는 '텐서 크기'가 아니라 '파이썬에서 띄우는 커널/동기화 횟수'였고, 다음 3가지를
  배칭해 제거했다 (분석: claude_analysis/corruption_cuda_acceleration_analysis.md):

    1) 레벨별 배치 FK (PenetrationEvaluator) — 스켈레톤은 트리이므로 '같은 깊이의 본은 서로
       독립'이다. 본 21개 파이썬 루프 → 트리 깊이(≤5) 루프로 줄이고, 페어별 캡슐 거리도
       인덱스 텐서로 한 번에 계산한다. 참조되지 않는 본은 FK에서 제외(가지치기)한다.
       ⚠️ PARENTS/BONE_NAMES(21본)는 데이터 계약(quats_84.reshape(...,21,4))이므로 절대
          줄이지 않는다. 가지치기는 'FK 루프가 방문할 본'만 제한한다.
    2) _seg_stats 벡터화 — 후보마다 float()를 3번 부르며 동기화하던 것을 [M,3] 텐서 1회
       전송으로 대체. (CPU에서도 이득이지만 GPU에서는 이것이 없으면 되려 느려진다.)
    3) 교차-샘플 후보 생성 (_build_candidates) — 샘플마다 후보를 만들던 호출을 라운드당
       1회로 합친다. 인덱스/각도 배열은 '호스트에서 조립해 한 번에 전송'해야 한다
       (디바이스에서 원소별로 채우면 전송이 M회 발생해 오히려 느려진다).

  알고리즘·난수 소비 순서·판정 기준은 개편 전과 완전히 동일하며, CPU 경로는 개편 전 코드와
  '비트 단위로 동일한' 결과를 낸다(회귀 게이트로 검증). 즉 이 개편은 기존 체크포인트/CSV와의
  비교 가능성을 훼손하지 않는다.

  디바이스: 기본은 CPU(비트 동일 보장). 윈도우 텐서를 CUDA에 올려 호출하면 그대로 GPU에서
  평가되며 더 빨라지지만, 부동소수점 차이(~1e-7)로 비트 동일은 보장되지 않는다.
"""
import math
import weakref

import numpy as np
import torch

from dataset_pipeline import BONE_MAP, BONE_NAMES, PARENTS

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


# ------------------------------------------------------------------
# 침투 깊이 평가기 — 레벨별 배치 FK + 페어 동시 캡슐 거리
# ------------------------------------------------------------------
class PenetrationEvaluator:
    """
    고정 페어 집합에 대한 침투 깊이 평가기. 인덱스·오프셋·임계값을 생성 시점에 한 번만
    준비해 두고, 이후에는 후보 묶음 [M, S, 87]을 통째로 받아 프레임별 깊이 [M, S]를 낸다.

    physics_module.DifferentiablePhysics와 '수치적으로 동일'하되 다음이 다르다:
      - FK가 본 21개 파이썬 루프 대신 '트리 깊이별 배치'로 돈다 (커널 런치 21회 → ≤5회).
        같은 깊이의 본은 부모가 이미 확정되어 서로 독립이므로 순서를 바꿔도 각 원소의
        연산 자체는 동일하다 → 비트 단위로 같은 결과가 나온다.
      - 페어 집합이 참조하지 않는 본(및 그 조상이 아닌 본)은 아예 계산하지 않는다.
        4페어 기준 21본 → 17본.
      - 페어별 캡슐 거리를 루프 대신 인덱스 텐서 하나로 동시에 구한다.

    ⚠️ 페어 집합에 종속적이다(인덱스·가지치기가 사전 계산됨). 다른 페어 집합에는
       반드시 별도 인스턴스를 쓸 것 — 재사용하면 결과가 조용히 달라진다.
       (모듈 내부에서는 _get_evaluator가 (physics, pairs, device)별로 캐시한다.)
    """

    def __init__(self, physics, pairs, device='cpu', dtype=torch.float32, prune=True):
        self.device = torch.device(device)
        self.dtype = dtype
        self.n_bones = len(BONE_NAMES)
        self.pairs = tuple(tuple(map(tuple, p)) for p in pairs)

        # --- FK에서 실제로 필요한 본만 남긴다 (페어가 쓰는 본 + 그 조상 전체) ---
        need = set()
        for (a, b), (c, d) in pairs:
            need |= {a, b, c, d}
        if prune:
            keep = set()
            for bone in need:
                cur = bone
                while cur is not None:
                    keep.add(cur)
                    cur = PARENTS[cur]
        else:
            keep = set(BONE_NAMES)

        # --- 본을 트리 깊이별로 묶는다 (같은 깊이 = 동시 계산 가능) ---
        depth = {}

        def _depth_of(b):
            if b not in depth:
                p = PARENTS[b]
                depth[b] = 0 if p is None else _depth_of(p) + 1
            return depth[b]

        for b in BONE_NAMES:
            _depth_of(b)

        self.levels = []
        for lv in range(1, max(depth[b] for b in keep) + 1):
            bones_at_lv = [b for b in BONE_NAMES if depth[b] == lv and b in keep]
            if not bones_at_lv:
                continue
            self.levels.append((
                torch.tensor([BONE_MAP[b] for b in bones_at_lv],
                             dtype=torch.long, device=self.device),
                torch.tensor([BONE_MAP[PARENTS[b]] for b in bones_at_lv],
                             dtype=torch.long, device=self.device),
                torch.stack([physics.bone_offsets[b] for b in bones_at_lv]).to(self.device, dtype),
            ))

        self.hips_i = BONE_MAP['Hips']

        # --- 페어별 캡슐 끝점 인덱스와 임계값(반지름 합) ---
        self.p1i = torch.tensor([BONE_MAP[p[0][0]] for p in pairs], dtype=torch.long, device=self.device)
        self.q1i = torch.tensor([BONE_MAP[p[0][1]] for p in pairs], dtype=torch.long, device=self.device)
        self.p2i = torch.tensor([BONE_MAP[p[1][0]] for p in pairs], dtype=torch.long, device=self.device)
        self.q2i = torch.tensor([BONE_MAP[p[1][1]] for p in pairs], dtype=torch.long, device=self.device)
        self.thr = torch.tensor([physics.bone_radii[p[0][1]] + physics.bone_radii[p[1][1]]
                                 for p in pairs], dtype=dtype, device=self.device)

    # -- 쿼터니언 연산 (physics_module과 동일 규약; unbind는 인덱싱과 수치 동일) --
    @staticmethod
    def _qmul(q1, q2):
        x1, y1, z1, w1 = q1.unbind(-1)
        x2, y2, z2, w2 = q2.unbind(-1)
        return torch.stack([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], dim=-1)

    @staticmethod
    def _qrot(q, v):
        q_xyz = q[..., :3]
        t = 2.0 * torch.cross(q_xyz, v, dim=-1)
        return v + q[..., 3:4] * t + torch.cross(q_xyz, t, dim=-1)

    def _capsule_distance(self, p1, q1, p2, q2):
        """Lumelsky 선분-선분 최단 거리. physics_module.capsule_distance의 배치판."""
        SMALL_NUM = 1e-8
        u = q1 - p1
        v = q2 - p2
        w = p1 - p2
        a = (u * u).sum(-1)
        b = (u * v).sum(-1)
        c = (v * v).sum(-1)
        d = (u * w).sum(-1)
        e = (v * w).sum(-1)
        D = a * c - b * b
        sD, tD = D, D

        degenerate = D < SMALL_NUM
        sN = torch.where(degenerate, torch.zeros_like(D), b * e - c * d)
        tN = torch.where(degenerate, e, a * e - b * d)
        tD = torch.where(degenerate, c, tD)

        m = sN < 0.0
        sN = torch.where(m, torch.zeros_like(sN), sN)
        tN = torch.where(m, e, tN)
        tD = torch.where(m, c, tD)

        m = sN > sD
        sN = torch.where(m, sD, sN)
        tN = torch.where(m, e + b, tN)
        tD = torch.where(m, c, tD)

        zeros = torch.zeros_like(a)

        m = tN < 0.0
        tN = torch.where(m, torch.zeros_like(tN), tN)
        sN = torch.where(m, torch.clamp(-d, min=zeros, max=a), sN)
        sD = torch.where(m, a, sD)

        m = tN > tD
        tN = torch.where(m, tD, tN)
        sN = torch.where(m, torch.clamp(-d + b, min=zeros, max=a), sN)
        sD = torch.where(m, a, sD)

        sc = torch.where(sN.abs() < SMALL_NUM, torch.zeros_like(sN),
                         sN / torch.clamp(sD, min=SMALL_NUM)).unsqueeze(-1)
        tc = torch.where(tN.abs() < SMALL_NUM, torch.zeros_like(tN),
                         tN / torch.clamp(tD, min=SMALL_NUM)).unsqueeze(-1)

        dP = w + (sc * u) - (tc * v)
        return torch.sqrt((dP * dP).sum(-1) + 1e-8)

    def frame_depths(self, windows):
        """
        후보 윈도우 묶음 [M, S, 87] → '프레임별' 최대 선형 관통 깊이(cm) [M, S].
        라운드당 이 호출 1회로 모든 후보를 평가하는 것이 이 모듈의 속도 핵심이다.
        """
        with torch.no_grad():
            M, S = windows.shape[0], windows.shape[1]
            local_q = windows[..., 3:].reshape(M, S, self.n_bones, 4)
            g_rot = torch.zeros(M, S, self.n_bones, 4,
                                device=windows.device, dtype=windows.dtype)
            g_pos = torch.zeros(M, S, self.n_bones, 3,
                                device=windows.device, dtype=windows.dtype)
            g_rot[:, :, self.hips_i] = local_q[:, :, self.hips_i]
            g_pos[:, :, self.hips_i] = windows[..., :3]

            # 트리 깊이별로 한 번에: 같은 레벨의 본은 부모가 이미 확정되어 서로 독립이다
            for child_i, parent_i, offset in self.levels:
                parent_rot = g_rot[:, :, parent_i]
                g_rot[:, :, child_i] = self._qmul(parent_rot, local_q[:, :, child_i])
                g_pos[:, :, child_i] = g_pos[:, :, parent_i] + \
                    self._qrot(parent_rot, offset.expand(M, S, -1, -1))

            dist = self._capsule_distance(g_pos[:, :, self.p1i], g_pos[:, :, self.q1i],
                                          g_pos[:, :, self.p2i], g_pos[:, :, self.q2i])
            return (torch.relu(self.thr - dist) * 100.0).amax(dim=-1)      # [M, S]

    def window_depths(self, windows):
        """
        후보 윈도우 묶음 [M, S, 87] → 윈도우별 최대 선형 관통 깊이(cm) [M].

        ⚠️ 이 '윈도우 하나당 스칼라 하나' 요약은 transient(sin-ramp: 애초에 한 봉우리만
        깊게 뚫는 것이 정의)에만 적합하다. persistent는 구간 '내내' 관통해야 하므로
        max 하나로는 품질을 판정할 수 없다 — _seg_stats_batched를 쓴다. (2026-08-07)
        """
        return self.frame_depths(windows).amax(dim=-1)                     # [M]


# (physics, pairs, device)별 evaluator 캐시.
#   - 공개 API가 physics 객체를 받는 형태를 유지하기 위한 장치다(호출부 무변경).
#   - evaluate.py는 파일마다 단일 윈도우 진입점을 호출하므로 캐시가 없으면 306회 재생성된다.
#   - physics를 약한 참조(weak key)로 잡아 physics가 사라지면 캐시도 함께 사라진다.
_EVALUATOR_CACHE = weakref.WeakKeyDictionary()


def _get_evaluator(physics, pairs, device):
    """physics가 이미 PenetrationEvaluator면 그대로, DifferentiablePhysics면 캐시에서 꺼낸다."""
    if isinstance(physics, PenetrationEvaluator):
        return physics
    key = (tuple(tuple(map(tuple, p)) for p in pairs), str(device))
    per_physics = _EVALUATOR_CACHE.setdefault(physics, {})
    ev = per_physics.get(key)
    if ev is None:
        ev = PenetrationEvaluator(physics, pairs, device=device)
        per_physics[key] = ev
    return ev


def _seg_stats_batched(frame_depths, seg_lo, seg_hi):
    """
    프레임별 깊이 [M, S]와 후보별 주입 구간 [M], [M] → (coverage [M], 중앙값 [M], 최대 [M]).
      - coverage : 구간 프레임 중 실제로 관통(>0)한 비율. '지속성'의 척도.
      - 중앙값   : 전형적인 프레임의 침투 깊이. 목표 범위(1~4cm) 판정에 쓴다.
    윈도우 max 대신 이 둘을 보는 이유: 같은 각도라도 포즈가 프레임마다 달라 깊이가
    출렁이므로, max만 보면 '한 프레임만 3cm 뚫린' 후보가 채택되어 나머지 프레임이
    무관통으로 남는다 (실측 2026-08-07: 주입 구간 프레임의 62%가 침투 0이었다).

    구판은 후보마다 float()를 3번 불러 동기화했다(배치당 ~6000회). 여기서는 전 후보를
    한 텐서로 처리하고 호출 측이 .cpu()를 1회만 하도록 만든다 — GPU 경로에서는 이 차이가
    전체 시간의 77%를 좌우한다(분석 §2-3).
    중앙값은 torch.median의 규약(짝수 길이 = 아래쪽 중앙값)을 그대로 재현한다.
    """
    M, S = frame_depths.shape
    ar = torch.arange(S, device=frame_depths.device).unsqueeze(0)            # [1, S]
    mask = (ar >= seg_lo.unsqueeze(1)) & (ar <= seg_hi.unsqueeze(1))         # [M, S]
    n = mask.sum(1)                                                          # [M]
    cov = (frame_depths > 1e-4).logical_and(mask).sum(1).to(frame_depths.dtype) / n.to(frame_depths.dtype)
    mx = frame_depths.masked_fill(~mask, float('-inf')).amax(1)
    srt = frame_depths.masked_fill(~mask, float('inf')).sort(dim=1).values
    med = srt.gather(1, ((n - 1) // 2).unsqueeze(1)).squeeze(1)
    return cov, med, mx


def _build_candidates(src, specs, S):
    """
    후보 명세 목록을 하나의 텐서 [M, S, 87]로 조립한다.

    specs 원소 = (src행 인덱스, 본 인덱스, f0, f1, 축[3], 각도열(라디안, 길이 f1-f0+1)).
    구판은 '샘플마다' 후보 생성 함수를 불렀고(배치당 ~186회) 그 비용이 텐서 크기가 아니라
    호출 횟수에 비례했다. 여기서는 라운드 전체의 후보를 한 번에 만든다.

    ⚠️ 인덱스/각도 배열은 반드시 '호스트(numpy)에서 조립한 뒤 한 번에 전송'해야 한다.
       디바이스 텐서를 원소별로 채우면(ang[m, f0:f1+1] = ...) 전송이 M회 발생해
       구판보다 느려진다 (실측: 4.9ms → 70ms).

    구간 밖 프레임은 각도 0 → delta = 항등이지만, 정규화까지 건너뛰도록 where로 원본
    쿼터니언을 그대로 남긴다 (구판이 지정 프레임만 손대던 동작과 비트 단위로 일치).
    """
    M = len(specs)
    dev, dt = src.device, src.dtype

    src_i_h = np.fromiter((s[0] for s in specs), dtype=np.int64, count=M)
    bone_i_h = np.fromiter((s[1] for s in specs), dtype=np.int64, count=M)
    f0_h = np.fromiter((s[2] for s in specs), dtype=np.int64, count=M)
    f1_h = np.fromiter((s[3] for s in specs), dtype=np.int64, count=M)
    axis_h = np.array([s[4] for s in specs], dtype=np.float32)
    ang_h = np.zeros((M, S), dtype=np.float32)
    for m, s in enumerate(specs):
        ang_h[m, s[2]:s[3] + 1] = s[5]

    src_i = torch.from_numpy(src_i_h).to(dev)
    bone_i = torch.from_numpy(bone_i_h).to(dev)
    f0 = torch.from_numpy(f0_h).to(dev)
    f1 = torch.from_numpy(f1_h).to(dev)
    axis = torch.from_numpy(axis_h).to(dev, dt)
    ang = torch.from_numpy(ang_h).to(dev, dt)

    out = src[src_i].clone()                                                 # [M, S, 87]
    ar = torch.arange(S, device=dev)
    mask = (ar >= f0.unsqueeze(1)) & (ar <= f1.unsqueeze(1))                 # [M, S]

    half = ang * 0.5
    delta = torch.cat([axis.unsqueeze(1) * torch.sin(half).unsqueeze(-1),
                       torch.cos(half).unsqueeze(-1)], dim=-1)               # [M, S, 4]

    quats = out[..., 3:].reshape(M, S, len(BONE_NAMES), 4)
    idx = bone_i.view(M, 1, 1, 1).expand(M, S, 1, 4)
    q = quats.gather(2, idx).squeeze(2)                                      # [M, S, 4]
    qn = _quat_mul(q, delta)
    qn = qn / (qn.norm(dim=-1, keepdim=True) + 1e-8)
    qn = torch.where(mask.unsqueeze(-1), qn, q)                              # 구간 밖은 원본 유지
    out[..., 3:] = quats.scatter(2, idx, qn.unsqueeze(2)).reshape(M, S, len(BONE_NAMES) * 4)
    return out


# ------------------------------------------------------------------
# 일시적(transient) 주입 — sin-ramp (그룹 단위 배치 처리)
# ------------------------------------------------------------------
def _inject_transient_group(src, idxs, ev, cfg, rng, S):
    """
    src [B,S,87]에서 idxs가 가리키는 샘플들에 일시적 주입 → [(손상 윈도우, meta), ...].
    라운드마다 샘플별로 (타이밍, θ_peak, 본, 축±) 후보 2개를 만들고 배치 FK 1회로 깊이를
    평가한다. 최소 관통(transient_min_depth_cm) 미달이면 재추첨; 전부 실패하면 마지막
    추첨을 그대로 사용한다 (무충돌 섭동도 유효한 복원 샘플 — meta['collided']=False).
    """
    n = len(idxs)
    sub = src[torch.tensor(idxs, device=src.device, dtype=torch.long)]
    results = [None] * n
    last = [None] * n
    pending = list(range(n))

    for round_i in range(1, cfg['transient_max_tries'] + 1):
        if not pending:
            break
        specs, owners, cmetas = [], [], []
        for j in pending:
            dur = rng.randint(cfg['transient_dur_range'][0], cfg['transient_dur_range'][1])
            start = rng.randint(-(dur // 2), S - 1 - dur // 2)   # 경계 클리핑 허용
            theta_peak = rng.uniform(cfg['theta_range_deg'][0], cfg['theta_range_deg'][1])
            bone = rng.choice(cfg['inject_bones'])
            axis = _rand_unit_axis(rng)

            f0, f1 = max(0, start), min(S - 1, start + dur)
            angles = [math.radians(theta_peak) * math.sin(math.pi * (f - start) / dur)
                      for f in range(f0, f1 + 1)]

            # 축 방향이 몸 바깥쪽일 수 있으므로 ±axis 두 후보를 같은 라운드에 평가
            # (θ 분포 자체는 설계값 U[15°,70°] 유지; 방향만 보정)
            for ax in (axis, [-a for a in axis]):
                specs.append((j, BONE_MAP[bone], f0, f1, ax, angles))
                owners.append(j)
                cmetas.append(dict(type='transient', bone=bone, bone_idx=BONE_MAP[bone],
                                   frames=(f0, f1), theta_peak_deg=round(theta_peak, 2),
                                   axis=[round(a, 4) for a in ax], tries=round_i))

        cands = _build_candidates(sub, specs, S)
        depths = ev.window_depths(cands).cpu().tolist()          # 전송 1회

        accepted = set()
        for c_i in range(len(specs)):
            j = owners[c_i]
            if j in accepted or results[j] is not None:
                continue
            meta = dict(cmetas[c_i], max_depth_cm=round(depths[c_i], 3), collided=True)
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
    ev = _get_evaluator(physics, pairs, window.device)
    return _inject_transient_group(window.unsqueeze(0), [0], ev, cfg, rng, window.shape[0])[0]


# ------------------------------------------------------------------
# 지속적(persistent) 주입 — 깊이 목표(1~4cm) 각도 탐색 (그룹 단위 배치 처리)
# ------------------------------------------------------------------
def _inject_persistent_group(src, idxs, ev, cfg, rng, S):
    """
    src [B,S,87]에서 idxs가 가리키는 샘플들에 지속 주입 → [(손상 윈도우, meta), ...].
    샘플별로 (본, ±축)을 추첨하고 각도 사다리 전체를 후보로 만들어 배치 FK 1회로 깊이를
    평가한다. 사다리에 목표 범위(1~4cm)가 없으면 범위를 감싸는 두 각도 사이를 이분 보간
    (다음 라운드), 양방향 모두 무침투면 본/축 재추첨. persistent_max_rounds 안에 실패하면
    클린 폴백(meta['fallback']=True) — 호출 측이 카운트해 비율 왜곡을 감시.
    """
    n = len(idxs)
    dev = src.device
    sub = src[torch.tensor(idxs, device=dev, dtype=torch.long)]
    lo, hi = cfg['persistent_depth_range_cm']
    mid = (lo + hi) * 0.5
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
    pending = [dict(j=j, refine=None) for j in range(n)]

    for round_i in range(1, cfg['persistent_max_rounds'] + 1):
        if not pending:
            break
        specs, owners, cmetas = [], [], []
        for p in pending:
            j = p['j']
            f0, f1, seg = seg_info[j]
            n_frames = f1 - f0 + 1
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
            for bone, axes, angle_list in draws:
                axes = [axes] if isinstance(axes[0], float) else list(axes)
                for ax in axes:
                    for theta in angle_list:
                        specs.append((j, BONE_MAP[bone], f0, f1, ax,
                                      [math.radians(theta)] * n_frames))
                        owners.append(id(p))
                        cmetas.append(dict(bone=bone, axis=ax, theta=theta,
                                           frames=(f0, f1), seg=seg))

        cands = _build_candidates(sub, specs, S)
        frame_depths = ev.frame_depths(cands)                                  # [M, S]
        seg_lo = torch.tensor([s[2] for s in specs], device=dev, dtype=torch.long)
        seg_hi = torch.tensor([s[3] for s in specs], device=dev, dtype=torch.long)
        cov_t, med_t, mx_t = _seg_stats_batched(frame_depths, seg_lo, seg_hi)
        stats = torch.stack([cov_t, med_t, mx_t], dim=1).cpu().tolist()        # 전송 1회

        # 샘플별 후보 결과 취합. 깊이 판정은 '윈도우 max'가 아니라 주입 구간의
        # (coverage, 중앙값)으로 한다 — 지속형의 정의가 '구간 내내 관통'이기 때문.
        by_p = {id(p): [] for p in pending}
        for c_i in range(len(specs)):
            cov, med, mx = stats[c_i]
            by_p[owners[c_i]].append((c_i, cmetas[c_i], cov, med, mx))

        still = []
        for p in pending:
            j = p['j']
            entries = by_p[id(p)]
            # 1) coverage를 만족하면서 깊이 중앙값이 목표 범위인 후보 → 중앙에 가장 가까운 것
            ok = [e for e in entries if e[2] >= min_cov and lo <= e[3] <= hi]
            if ok:
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
            results[j] = (sub[j].clone(), dict(type='clean', fallback=True, seg=seg))
    return results


def inject_persistent(window, physics, cfg, rng, pairs):
    """단일 윈도우 [S,87] 진입점 (evaluate.py 등에서 사용). 원본은 보존."""
    ev = _get_evaluator(physics, pairs, window.device)
    return _inject_persistent_group(window.unsqueeze(0), [0], ev, cfg, rng, window.shape[0])[0]


# ------------------------------------------------------------------
# 배치 단위 진입점 (train.py가 매 배치 호출)
# ------------------------------------------------------------------
def corrupt_batch(clean_batch, physics, cfg, rng, pairs):
    """
    clean_batch [B, S, 87] → (모델 입력용 손상 배치, 샘플별 meta 리스트).
    타겟은 항상 원본 clean_batch (호출 측이 그대로 보관).
    혼합비: clean_ratio는 전체 대비, transient_ratio는 '손상 샘플' 중 비율.
      (기본 0.5 / 0.3 → 전체 분포: 클린 50% / 일시적 15% / 지속적 35%)

    physics 인자는 DifferentiablePhysics 또는 PenetrationEvaluator 둘 다 받는다.
    전자를 넘기면 (physics, pairs, device)별 evaluator를 내부에서 캐시해 재사용한다 —
    호출부를 바꾸지 않고도 배치 경로를 쓰기 위한 장치다.
    """
    B, S = clean_batch.shape[0], clean_batch.shape[1]
    ev = _get_evaluator(physics, pairs, clean_batch.device)
    out = clean_batch.clone()
    metas = [dict(type='clean') for _ in range(B)]

    tr_idx, pe_idx = [], []
    for i in range(B):
        if rng.random() < cfg['clean_ratio']:
            continue
        (tr_idx if rng.random() < cfg['transient_ratio'] else pe_idx).append(i)

    if tr_idx:
        for i, (w, m) in zip(tr_idx, _inject_transient_group(
                clean_batch, tr_idx, ev, cfg, rng, S)):
            out[i] = w
            metas[i] = m
    if pe_idx:
        for i, (w, m) in zip(pe_idx, _inject_persistent_group(
                clean_batch, pe_idx, ev, cfg, rng, S)):
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

    목적: 주입이 CPU에서 도는 동안 GPU가 노는 것을 막는다. GPU가 배치 N을 학습하는 동안
    배치 N+1의 주입을 준비해 주입 비용을 GPU 시간 뒤로 숨긴다 (에폭 ≈ max(주입, GPU) + ε).

    ⚠️ 주입을 GPU에서 수행하면(윈도우 텐서를 CUDA에 올려 호출) 학습 스텝과 같은 스트림을
       쓰므로 겹칠 것이 없어 이 프리페치는 이득이 아니라 손해가 된다(실측 1.6배). GPU 경로를
       택할 때는 이 제너레이터 대신 corrupt_batch를 직접 부를 것.

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
