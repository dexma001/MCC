"""미분가능 사영 레이어 (Projection Layer) — R1.5-3 / 백로그 C1.

[이 모듈이 존재하는 이유]
  MSE와 소프트 물리 페널티로 학습한 네트워크는 관통의 '기댓값'을 줄일 뿐,
  '상한'을 강제할 메커니즘이 없다. 드문 한 프레임의 9cm 관통은 배치 평균 안에서 값싸다.
  실측(evaluate_results.csv, 해부학 반지름 시대 λ 6점 스윕)이 이를 증명한다 —
  legacy80/transient의 max_pen4_after가 λ 0.3~0.7 전 구간에서 8.2~8.9cm에 붙어 있고,
  주입 전 값(8.83~8.87cm)과 사실상 같다. 즉 모델은 '넓이'(제거율 77~89%)는 잡지만
  '깊이'(꼬리)는 전혀 못 잡는다. 이것은 튜닝 부족이 아니라 소프트 제약의 구조적 천장이다.

  이 레이어는 네트워크 출력 뒤에서 기하학적으로 관통을 밀어내어
  `collision <= eps` 를 희망에서 준(準)보장으로 바꾼다.

[알고리즘 — 페널티 경사하강이 아니라 '선형화 제약 사영'(PBD 계열)]
  페어 j의 관통 깊이를 d_j(q) = relu(thr_j + margin - dist_j(FK(hips, q))) 라 할 때,
  풀고 싶은 문제는

      q* = argmin ||q - q_out||^2   s.t.  d_j(q) = 0  for all j

  이고, 이를 1차 선형화 + 최소노름 해로 반복한다 (g_j = d d_j / d q):

      delta_j = -(d_j / ||g_j||^2) * g_j
      q <- normalize(q + omega * sum_j delta_j)

  이 형태를 고른 이유 3가지:
    1) 각 스텝이 이미 '최소노름 해'이므로, 요구사항이 필수라고 못박은 ||q - q_out||^2
       최소이동 항이 별도 가중치 없이 구조에 내장된다 (튜닝 상수 1개 감소).
       단, 이는 스텝별 국소 최소이지 전역 최소가 아니다.
    2) d/||g||^2 이 "이 방향으로 정확히 d만큼 밀어내는 거리"라는 물리적 의미를 가져
       스텝 크기 eta를 튜닝할 필요가 없다 (페널티 GD는 필요하다).
    3) 위반이 없으면 루프가 즉시 break 하므로 클린 입력에 대해 '비트 단위 항등'이다.
       어떤 lambda 로도 얻지 못한 do-no-harm 보장이다 (현 v1은 클린 입력에서
       max_pen 1.75 -> 2.46cm 로 없던 충돌을 만든다).

[성능 설계 — 실측으로 정해진 것]
  * 위반 프레임 마스킹이 유일하게 유효한 최적화다. 클린 시나리오는 위반 프레임이 0개라
    검출 FK 비용만으로 끝난다 (0.10 ms/frame).
  * 페어 사본을 배치로 쌓아 backward 횟수를 줄이는 최적화는 '더 느리다'
    (FK가 배치 크기에 비례해 커져 절약분을 잡아먹는다). 시도하지 말 것.
  * 30프레임 윈도우는 프레임끼리 독립이므로 grad(sum_f d_j[f]) 한 번이 프레임별
    그래디언트를 전부 준다. backward 횟수는 프레임 수가 아니라 '활성 페어 수'에 비례한다.
  * 단일 윈도우 추론에서 CUDA는 CPU보다 3~6배 느리다 (작은 텐서 커널 런치 오버헤드).
    추론은 CPU, 학습 배치는 GPU 일괄이 올바른 배치다.

[파이프라인 순서 — 중요]
  모델 -> (저역통과 필터) -> 사영.  이 순서만이 `collision <= eps` 를 유지한다.
  필터를 사영 뒤에 두면 저역통과가 사영 결과를 뭉개 관통을 되살린다
  (실측: persistent max_pen4 0.138 -> 4.939cm).
"""

import time

import torch


# =====================================================================
# [조절 손잡이] 전부 모듈 상수 — 하드코딩 금지, run_config/CSV에 기록된다.
# =====================================================================
PROJ_ENABLED = True
"""사영 레이어 on/off.

[2026-08-17] False -> True. 회귀 게이트(V1)를 먼저 통과시킨 뒤 켰다:
  PROJ_ENABLED=False 상태로 306파일 × 4시나리오를 돌려 기존 50열 184셀이
  도입 전 기준선과 **불일치 0**임을 확인했다. 즉 이 스위치를 다시 False로 되돌리면
  사영 도입 이전과 완전히 같은 결과가 나온다 (가장 값싼 롤백 수단).

[주의] 이 기본값이 '회귀 게이트'를 성립시킨다 — False일 때 evaluate.py는 사영을 호출조차
   하지 않으므로, 기존 50열 지표가 도입 전과 수치까지 동일해야 한다. 그 동일성이
   확인된 뒤에만 True로 바꾼다."""

PROJ_K = 8
"""반복 횟수 상한.

[주의] '비용이 K에 비례한다'는 직관은 틀렸다 — 수렴한 프레임은 위반 마스크에서 빠지고,
   전부 빠지면 루프가 break 하므로 **수렴 이후의 K는 사실상 공짜다.**
   실측(동결 v1의 '모델 출력', omega=1.8): K=4 -> 0.366 ms/frame, K=8 -> 0.389 ms/frame.
   6% 더 비싼 대신 잔존이 0.180 -> 0.000cm 가 된다.

실측 근거 (동결 v1 모델 출력 60파일 × 4시나리오, 사영 후 4페어 최악 관통 cm):
  omega=1.0 K=4 : clean 0.000 / legacy80 1.115 / transient 1.276 / persistent 0.000
  omega=1.0 K=8 : 0.000 / 0.233 / 0.389 / 0.000
  omega=1.8 K=4 : 0.000 / 0.180 / 0.427 / 0.000
  omega=1.8 K=8 : 0.000 / 0.000 / 0.000 / 0.000   <-- 채택
[주의] 이 값들은 '주입 원본'이 아니라 '모델 출력'에 대한 것이다. 모델이 이미 대부분을
   제거해 위반 프레임이 2% 남짓이므로, 주입 원본으로 잰 초기 벤치(K=4에서 0.565cm 등)보다
   훨씬 쉬운 문제다. 운용점은 반드시 실제 조건에서 골라야 한다."""

PROJ_OMEGA = 1.8
"""과이완(over-relaxation) 계수. 수렴 속도와 이동량의 교환 손잡이.

[최소이동 성질의 정직한 서술 — 단위 검증 V3-b가 실측한 것]
  케이스마다 lr x steps 35조합을 전탐색해 '같은 잔존에 도달하면서 이동이 최소'인
  페널티 경사하강(= 오라클 튜닝된 대조군)과 비교하면:
      omega=1.0 : 사영 0.2776cm vs GD 0.3792cm  -> 비 0.732 (27% 적게 움직인다)
      omega=1.8 : 사영 0.3912cm vs GD 0.3833cm  -> 비 1.021 (2.1% 더 움직인다)
  즉 **최소노름 성질은 omega=1.0에서만 엄밀히 성립한다.** omega=1.8은 의도적 오버슈트다.

[그럼에도 1.8을 기본값으로 둔 이유]
  실제 조건(모델 출력)에서는 위반 프레임이 2% 남짓이라 두 설정의 이동량 차이가
  0.0107cm vs 0.0158cm = **0.005cm(0.05mm)** 에 불과하다. 모델 자신의 교정량(약 3.2cm)의
  0.2% 수준이라 의미가 없는 반면, omega=1.8은 K=8에서 잔존을 0.000cm로 완전히 없앤다.
  ⇒ 이 작업에서는 '완전 제거'가 '0.05mm 덜 움직이기'보다 가치가 크다.
  주입 원본처럼 위반 프레임이 80%인 상황을 다루게 되면 이 판단은 재검토해야 한다.

[주의] 2.0 이상은 진동/발산 위험. 상한으로 취급할 것."""

PROJ_MARGIN_CM = 0.2
"""안전 여유 [cm]. '교정 목표'를 depth <= 0 이 아니라 depth <= -margin 으로 둔다.

경계에 정확히 붙여 놓으면 이후 어떤 연산(재정규화 오차, 필터)에도 다시 관통으로
넘어가기 쉽다. 또 relu의 꺾임 지점에서 프레임 간 진동이 생긴다.

[주의] 이것은 '교정 목표'이지 '검출 기준'이 아니다 — 아래 PROJ_DETECT_MARGIN_CM 참조."""

PROJ_DETECT_MARGIN_CM = 0.0
"""'어느 프레임을 손댈 것인가'의 판정 기준 [cm]. 교정 목표(PROJ_MARGIN_CM)와 분리한다.

이 분리가 없으면(검출도 margin으로 하면) **관통이 없는데 0.2cm 이내로 근접한 프레임까지
사영이 건드려, 이 레이어의 핵심 보장인 '클린 입력 = 비트 단위 항등'이 깨진다.**
단위 검증 V3-a'가 실제로 이것을 잡아냈다.

  검출: depth(margin=0) > 0  → '진짜 관통한' 프레임만 대상
  교정: 그 프레임을 depth <= -PROJ_MARGIN_CM 까지 밀어냄 → 여유는 확보

부작용: 관통 0.05cm 프레임은 -0.2cm까지 밀리는데 -0.05cm 프레임은 그대로 둔다(불연속).
그러나 '손댈 이유가 없는 입력은 한 비트도 바꾸지 않는다'는 do-no-harm 보장이
그 균일성보다 가치가 크다고 판단했다."""

PROJ_DAMP = 1e-6
"""||g||^2 이 0에 가까운 퇴화 구성(거의 평행한 두 캡슐)에서의 감쇠항."""

PROJ_PAIRS = "trained4"
"""사영 대상 페어 집합. "trained4" = train.COLLIDING_PAIRS (학습이 실제 최적화하는 4쌍).

"all112"는 실측으로 기각됐다: 3.45 ms/frame(예산 초과)인 데다 제약끼리 경쟁해
4페어 잔존이 오히려 나빠진다(0.565 -> 1.892cm). 전신 보장은 페어 프루닝(D1)이
선행돼야 하는 별건이며, v1이 보장하는 것은 '학습된 4쌍에 대한 collision <= eps'다.
(참고: 4쌍 사영 후에도 112페어 최악 관통은 ~5cm 남는다. 8.19cm에서 개선은 되지만
 전신 보장이 아니라는 뜻이다.)"""


# =====================================================================
# 내부 헬퍼
# =====================================================================
def _pair_thresholds(physics, pairs):
    """페어별 접촉 임계값(두 자식 본 반지름의 합) [m] 리스트.

    physics_module.get_collision_loss / get_penetration_depths 와 동일한 규약:
    임계값은 '자식 본'(캡슐의 끝점)의 반지름 합이다.
    """
    return [physics.bone_radii[c1] + physics.bone_radii[c2]
            for (_p1, c1), (_p2, c2) in pairs]


def _depths_with_margin(physics, hips, quats, pairs, thresholds, margin_m):
    """[F, n_pairs] 여유를 포함한 관통 깊이 relu(thr + margin - dist).

    physics_module.get_penetration_depths 를 그대로 쓰지 못하는 이유:
    margin 은 relu '안'으로 들어가야 하므로(relu(t+m-d) != relu(t-d)+m)
    거리 계산만 공유하고 임계 비교는 여기서 한다. 거리 자체는 physics의
    capsule_distance/forward_kinematics를 그대로 재사용한다 (물리 코드 중복 금지).
    """
    gp = physics.forward_kinematics(hips, quats)
    out = []
    for ((p1, c1), (p2, c2)), thr in zip(pairs, thresholds):
        dist = physics.capsule_distance(gp[p1], gp[c1], gp[p2], gp[c2])
        out.append(torch.relu(thr + margin_m - dist))
    return torch.stack(out, dim=-1)


def _normalize_quats(flat84):
    """[N, 84] -> 관절별(21개) 단위 쿼터니언으로 재정규화. models.py:78 과 동일 규약."""
    q = flat84.view(-1, 21, 4)
    return (q / (q.norm(dim=-1, keepdim=True) + 1e-8)).view(-1, 84)


# =====================================================================
# 공개 API
# =====================================================================
def project_window(physics, hips, quats, pairs,
                   k=None, omega=None, margin_cm=None, damp=None,
                   detect_margin_cm=None, collect_stats=True):
    """관통이 남은 프레임을 제약면 위로 밀어낸다.

    hips  : [F, 3]  루트 위치. 읽기만 하며 절대 수정하지 않는다 (교정 대상이 아닌 통과값).
    quats : [F, 84] 모델이 출력한 21관절 로컬 쿼터니언.
    pairs : 충돌 페어 목록 (train.COLLIDING_PAIRS 규약).

    반환: (quats_projected [F, 84], stats dict)
      stats = {
        "iters_used"       : 실제로 돈 반복 횟수 (조기 종료 포함),
        "frames_touched"   : 한 번이라도 수정된 프레임 수,
        "frames_total"     : F,
        "residual_max_cm"  : 사영 후 4페어 최악 관통 (margin 제외한 '진짜' 관통),
        "move_cm"          : 사영이 유발한 평균 관절 위치 이동 (collect_stats=True일 때만),
        "ms"               : 소요 시간 [ms],
      }

    보장:
      * 위반 프레임이 하나도 없으면 입력과 '비트 단위로 동일한' 텐서를 돌려준다.
      * 출력은 항상 관절별 단위 쿼터니언이다.
      * hips는 변경되지 않는다 (애초에 반환값에 포함하지 않는다).
      * 난수를 쓰지 않으므로 같은 입력에 항상 같은 출력 = 결정론적이다.
    """
    k = PROJ_K if k is None else k
    omega = PROJ_OMEGA if omega is None else omega
    margin_m = (PROJ_MARGIN_CM if margin_cm is None else margin_cm) / 100.0
    detect_m = (PROJ_DETECT_MARGIN_CM if detect_margin_cm is None else detect_margin_cm) / 100.0
    damp = PROJ_DAMP if damp is None else damp

    t0 = time.perf_counter()
    thresholds = _pair_thresholds(physics, pairs)

    q = quats
    touched = torch.zeros(quats.shape[0], dtype=torch.bool, device=quats.device)
    iters_used = 0

    for _ in range(max(0, k)):
        # 1) 검출 — 그래디언트 불필요. '진짜 관통한'(detect_m 기준) 프레임만 고른다.
        #    교정 목표(margin_m)와 분리해야 무관통 프레임에 대한 항등이 보장된다.
        with torch.no_grad():
            d0 = _depths_with_margin(physics, hips, q, pairs, thresholds, detect_m)
        viol = (d0.max(dim=-1).values > 0).nonzero(as_tuple=True)[0]
        if viol.numel() == 0:
            break   # ← 클린 입력이면 여기서 즉시 끝난다: q는 입력 객체 그대로다.

        iters_used += 1
        touched[viol] = True

        # 2) 위반 프레임만 모아 한 번의 FK로 그래디언트 경로를 만든다.
        #    [주의] enable_grad 필수 — 호출부(evaluate.py의 파일 루프)가 `with torch.no_grad():`
        #       안이라 이것이 없으면 grad_fn이 기록되지 않아 autograd.grad가 터진다.
        #       (클린 시나리오는 위반 프레임이 0개라 이 경로에 진입조차 하지 않으므로
        #        스모크 테스트에서 조용히 통과한다 — 실제로 그렇게 놓쳤다.)
        with torch.enable_grad():
            qs = q[viol].detach().requires_grad_(True)
            gp = physics.forward_kinematics(hips[viol], qs)

            delta = torch.zeros_like(qs)
            for ((p1, c1), (p2, c2)), thr in zip(pairs, thresholds):
                dist = physics.capsule_distance(gp[p1], gp[c1], gp[p2], gp[c2])
                d = torch.relu(thr + margin_m - dist)
                if float(d.max()) <= 0.0:
                    continue    # 비활성 페어는 backward 자체를 건너뛴다 (비용 절감의 핵심)
                # 프레임끼리 독립이므로 sum 의 그래디언트가 곧 프레임별 그래디언트다.
                g = torch.autograd.grad(d.sum(), qs, retain_graph=True)[0]
                gg = (g * g).sum(dim=-1, keepdim=True)
                # 선형화 최소노름 스텝: 이 방향으로 정확히 d 만큼 밀어낸다.
                delta = delta - omega * (d.detach().unsqueeze(-1) / (gg + damp)) * g

        # 3) 적용 + 재정규화. in-place 대신 clone 으로 호출자 텐서를 보호한다.
        q = q.clone()
        q[viol] = _normalize_quats(qs.detach() + delta.detach())

    q = q.detach()

    stats = {
        "iters_used": iters_used,
        "frames_touched": int(touched.sum()),
        "frames_total": int(quats.shape[0]),
        "residual_max_cm": 0.0,
        "move_cm": 0.0,
        "ms": 0.0,
    }
    if collect_stats:
        with torch.no_grad():
            # margin 을 뺀 '진짜' 관통으로 잔존을 보고한다 (evaluate 의 max_pen4 와 같은 정의).
            res = _depths_with_margin(physics, hips, q, pairs, thresholds, 0.0)
            stats["residual_max_cm"] = float(res.max()) * 100.0
            if iters_used > 0:
                gp_in = physics.compute_global_pos_tensor(hips, quats)
                gp_out = physics.compute_global_pos_tensor(hips, q)
                stats["move_cm"] = float((gp_out - gp_in).norm(dim=-1).mean()) * 100.0
    stats["ms"] = (time.perf_counter() - t0) * 1000.0
    return q, stats
