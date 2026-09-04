"""저역통과 레이어 (Low-Pass Layer) — R1.5-4 / 백로그 C2.

[계획서와 이름이 다른 점 — 먼저 밝힌다]
  설계 계획 `claude_analysis/lowpass_layer_plan.md` §4.1/§4.3은 이 모듈을
  `smoothing.py` / 상수 `SMOOTH_*` / CSV 열 `smooth_*` 로 적어두었다.
  사용자 지시로 파일명을 `lowpass.py` 로 정했고, 접두어도 파일명에 맞춰 `LP_*` / `lp_*` 로
  통일했다. **바뀐 것은 이름뿐이고 §4.1의 구조(D1~D5), §6의 수용 기준, §7의 검증 항목은
  그대로다.**

[이 모듈이 존재하는 이유]
  모델은 프레임별 delta 를 '독립적으로' 내놓는 잔차 구조이고, 손실 항 중 어느 것도
  시간축을 소유하지 않는다 (KL은 제거됐고 물리 손실은 프레임 단위다).
  그 결과 지터(mean||accel||)가 전 20행·전 설정·전 시나리오에서 예외 없이 1.9~3.8배로
  악화된다. anat 스윕에서 jitter_after 는 lambda 에 대해 비단조이므로
  (0.3: 1.28 / 0.45: 1.34 / 0.5: 1.01) **lambda 튜닝으로는 제거할 수 없고,
  '최소 지점 선택'까지만 가능하다.** 후처리 필터가 필요한 이유가 이것이다.

[지터의 정체 — 설계를 좌우한 측정 (계획서 §1.2)]
  총 가속도 에너지는 변하지 않는다 (출력/GT = 1.00, 1~15Hz 전 bin). 그런데 지표
  mean||accel|| 은 2.2배다. 두 통계가 다른 것을 보기 때문이다:
    총에너지 = sum||a||^2  -> 큰 가속도 몇 개(실제 동작 전환)가 지배
    지표     = mean||a||   -> 작고 넓게 퍼진 성분이 지배
  꼬리 측정이 이를 확인한다: transient 의 '최대' 프레임 가속도는 입력 52.8 -> 모델 41.7 로
  오히려 준다. 즉 **모델은 큰 실제 가속도를 깎고 전 프레임·전 관절에 저진폭 트레머를
  고르게 퍼뜨린다.** 결함이 특정 관절에 몰려 있지 않으므로 전역 저역통과가 맞는 도구이고,
  관절별 적응 필터는 불필요하다. 목표는 지터 0 이 아니라 **GT 지터(약 0.42)로 복귀**다.

[왜 영위상(zero-phase)인가 — 계획서 §3 발견 2]
  같은 지터 수준(약 0.41)을 만드는 두 필터의 MPJPE 비용:
      MA5 영위상  : 3.138 -> 3.174  (+0.04cm)
      EMA 2Hz 인과: 3.138 -> 4.048  (+0.91cm)
  23배 차이다. 손해의 정체는 평활화가 아니라 **군지연**이고, 백본이 이미 비인과
  (30프레임 양방향 어텐션)이므로 윈도우 내 영위상 필터는 추가 지연이 0이다.
  인과 필터(one_euro)는 코드만 넣어두고 기본값에서 제외한다 — R2-1(causal 백본)이
  도착할 때 지연 예산과 함께 도입한다.

[파이프라인 순서 — 이 모듈 단독 배포 금지]
  모델 -> (이 필터) -> 사영.  **필터는 관통을 되살린다** (계획서 §3 발견 3:
  persistent max_pen4 0.138 -> 4.939cm, 자기 사전등록 가드레일 +0.5cm 를 10배 위반).
  경계에 딱 붙어 있던 포즈를 이웃 프레임과 평균 내면 경계 안쪽으로 밀려 들어가기 때문이다.
  따라서 이 레이어는 **반드시 projection.py 앞에** 놓여야 하며, 사영(C1) 없이 이것만
  켜는 구성은 금지다. 결합(모델->MA5->사영)은 두 항목의 사전등록 기준을 동시에 통과한다
  (지터 0.87~1.03배, max_pen4 0.00~0.16cm, MPJPE 악화 <= +0.04cm).

[결정론] 난수를 쓰지 않는다. 같은 입력 -> 같은 출력.
"""

import math
import time

import torch


# =====================================================================
# [조절 손잡이] 전부 모듈 상수 — 하드코딩 금지, CSV에 기록된다.
# =====================================================================
LP_ENABLED = False
"""저역통과 레이어 on/off.

[주의] 이 기본값 False 가 '회귀 게이트(V1)'를 성립시킨다 — False 일 때 evaluate.py 는
   이 모듈을 호출조차 하지 않으므로, 기존 60열 지표가 도입 전과 수치까지 동일해야 한다.
   그 동일성이 확인된 뒤에만 True 로 바꾼다 (사영 레이어에서 쓴 것과 같은 절차).
   가장 값싼 롤백 수단이기도 하다."""

LP_MODE = "ma_zerophase"
"""필터 형태. "ma_zerophase" | "savgol" | "one_euro".

  ma_zerophase : 중심 이동평균(대칭 FIR) = 군지연 0. v1.5 기본값.
  savgol       : Savitzky-Golay. 피크를 더 잘 보존한다 (계획서 §5 단계 A' 대조군).
  one_euro     : 인과 필터. **기본값에서 제외** — 위 docstring [왜 영위상인가] 참조.
                 v2 스트리밍(R2-1)에서 재사용하려고 자리만 만들어 둔 것이다."""

LP_WINDOW = 5
"""영위상 필터 창 크기 (홀수).

실측 근거 (계획서 §3 발견 1, 동결 v1 anat lambda0.5, held-out 40파일).
괄호는 jitter_before 대비 배수이고 사전등록 1차 기준은 <= 1.2배:
  MA3 : clean 1.16 / transient 1.02 / persistent 1.09 / legacy80 **1.23**  <- 탈락
  MA5 : clean 0.95 / transient 0.83 / persistent 0.89 / legacy80 0.99      <- 채택
MA5 는 전 시나리오에서 GT 지터(0.424) 수준으로 되돌리면서 MPJPE 비용이 +0.04cm 다."""

LP_EDGE = "replicate"
"""윈도우 경계 처리. "replicate" | "reflect" | "shrink".

  replicate : 첫/끝 프레임을 복제해 패딩 (기본값)
  reflect   : 거울 반사 패딩
  shrink    : 패딩 없이 '실재하는 표본만' 평균 (경계에서 창이 좁아진다)

[주의] 영위상 필터는 경계에서 패딩 가정에 의존한다 (계획서 §8 리스크 R3).
   30프레임 윈도우의 0번·29번 프레임은 어떤 모드를 쓰든 국소 오차가 커질 수 있고,
   이를 정량화하는 도구가 C3(경계 지표)다. 그때 이 상수를 함께 재검토한다."""

LP_SAVGOL_POLY = 2
"""Savitzky-Golay 다항식 차수. LP_MODE="savgol" 일 때만 쓰인다 (창 5, 차수 2 = 계획서 대조군)."""

LP_ONE_EURO_MIN_CUTOFF = 1.0
LP_ONE_EURO_BETA = 0.1
LP_ONE_EURO_D_CUTOFF = 1.0
"""OneEuro 계수. LP_MODE="one_euro" 일 때만 쓰인다.

실측(계획서 §3 발견 1): 1 Euro(fc=1, beta=0.1)는 지터를 0.68~0.82배까지 낮추지만
MPJPE 를 +1.45cm 악화시킨다 — 현재 백본에서는 손해다."""

FPS = 30
"""데이터셋 프레임률. one_euro 의 시간 상수 계산에만 쓰인다 (영위상 모드는 fps 무관)."""

_EPS = 1e-8


# =====================================================================
# 내부 헬퍼
# =====================================================================
def _hemispherize(q):
    """[F, 21, 4] 쿼터니언의 부호를 시간축으로 정렬한다. 반환 (정렬된 q, 부호 [F, 21]).

    [이 함수가 이 모듈 최대의 함정이다 — 계획서 D2]
    쿼터니언은 이중피복이라 q 와 -q 가 같은 회전이다. 인접 프레임의 부호가 뒤집힌 상태로
    성분별 평균을 내면 두 회전의 '중간'이 아니라 원점 근처의 쓰레기가 나온다 => 포즈 파괴.

    구현 메모: 정렬은 '이미 정렬된 이전 프레임'을 기준으로 해야 하므로 본질적으로 순차적이다.
    그러나 dot(q_t, flip_{t-1} * q_{t-1}) = flip_{t-1} * dot(q_t, q_{t-1}) 이므로
    flip_t = flip_{t-1} * sign(dot(q_t, q_{t-1})) 이고, 결국 **인접 내적 부호의 누적곱**이다.
    파이썬 루프 없이 cumprod 한 번으로 끝난다 (결정론적, 프레임 수에 대해 O(F)).
    """
    if q.shape[0] < 2:
        return q, torch.ones(q.shape[:2], dtype=q.dtype, device=q.device)
    dots = (q[1:] * q[:-1]).sum(dim=-1)                      # [F-1, 21]
    step = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
    flips = torch.cat([torch.ones_like(step[:1]), torch.cumprod(step, dim=0)], dim=0)   # [F, 21]
    return q * flips.unsqueeze(-1), flips


def _pad_time(x, pad, edge):
    """[F, D] 를 시간축 양끝으로 pad 만큼 확장. 반환 [F + 2*pad, D]."""
    if pad <= 0:
        return x
    if edge == "replicate":
        return torch.cat([x[:1].expand(pad, -1), x, x[-1:].expand(pad, -1)], dim=0)
    if edge == "reflect":
        # 경계 프레임 자신은 중복하지 않는 표준 반사 (x[pad], ..., x[1] | x | x[-2], ...)
        head = x[1:pad + 1].flip(0)
        tail = x[-pad - 1:-1].flip(0)
        return torch.cat([head, x, tail], dim=0)
    raise ValueError(f"알 수 없는 LP_EDGE: {edge}")


def _fir_zerophase(x, coeffs, edge):
    """[F, D] 에 대칭 FIR 계수를 중심 정렬로 적용한다 (군지연 0). 반환 [F, D].

    coeffs 는 길이 홀수의 1D 텐서이고 중심이 현재 프레임이다. 대칭 계수 + 중심 정렬이
    '영위상'의 정의 그 자체다 (필터를 두 번 거는 filtfilt 가 필요 없다).
    """
    f, d = x.shape
    w = coeffs.numel()
    pad = w // 2
    if edge == "shrink":
        # 패딩 없이 '실재하는 표본만' 쓰고 계수합으로 재정규화한다 (DC 이득 1.0 유지).
        zeros = torch.zeros(pad, d, dtype=x.dtype, device=x.device)
        xp = torch.cat([zeros, x, zeros], dim=0)
        mask = torch.cat([torch.zeros(pad, 1, dtype=x.dtype, device=x.device),
                          torch.ones(f, 1, dtype=x.dtype, device=x.device),
                          torch.zeros(pad, 1, dtype=x.dtype, device=x.device)], dim=0)
        num = torch.zeros_like(x)
        den = torch.zeros(f, 1, dtype=x.dtype, device=x.device)
        for i in range(w):
            num = num + coeffs[i] * xp[i:i + f]
            den = den + coeffs[i] * mask[i:i + f]
        return num / (den + _EPS)
    xp = _pad_time(x, pad, edge)
    out = torch.zeros_like(x)
    for i in range(w):
        out = out + coeffs[i] * xp[i:i + f]
    return out


def _ma_coeffs(window, dtype, device):
    """이동평균(boxcar) 계수. 합 1.0 => DC 이득 1.0."""
    return torch.full((window,), 1.0 / window, dtype=dtype, device=device)


def _savgol_coeffs(window, poly, dtype, device):
    """Savitzky-Golay 평활 계수 (미분 차수 0).

    오프셋 x_i = i - h 에 대한 Vandermonde A 를 최소제곱으로 풀면 중심값은 a_0 이고,
    a = pinv(A) y 이므로 계수는 **pinv(A) 의 0번 행**이다. scipy 의존성 없이 구한다.
    """
    h = window // 2
    xs = torch.arange(-h, h + 1, dtype=torch.float64, device=device)
    a = torch.stack([xs ** p for p in range(poly + 1)], dim=1)      # [window, poly+1]
    coeffs = torch.linalg.pinv(a)[0]                                # [window]
    return coeffs.to(dtype=dtype)


def _one_euro(x, fps, min_cutoff, beta, d_cutoff):
    """[F, D] 인과 OneEuro 필터. **기본 경로가 아니다** (v2 스트리밍용 자리).

    영위상 모드와 달리 군지연이 있으며, 그 지연이 그대로 MPJPE 비용으로 계상된다
    (실측 +1.45cm). 지금 켜지 말 것.
    """
    out = [x[0]]
    prev_x = x[0]
    prev_dx = torch.zeros_like(x[0])
    tau_d = 1.0 / (2.0 * math.pi * d_cutoff)
    a_d = 1.0 / (1.0 + tau_d * fps)
    for t in range(1, x.shape[0]):
        dx = (x[t] - prev_x) * fps
        dx_hat = a_d * dx + (1.0 - a_d) * prev_dx
        cutoff = min_cutoff + beta * dx_hat.abs()
        tau = 1.0 / (2.0 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau * fps)
        xhat = a * x[t] + (1.0 - a) * prev_x
        out.append(xhat)
        prev_x, prev_dx = xhat, dx_hat
    return torch.stack(out, dim=0)


def _normalize_quats(flat84):
    """[N, 84] -> 관절별(21개) 단위 쿼터니언으로 재정규화. models.py:78 / projection.py 와 동일 규약."""
    q = flat84.view(-1, 21, 4)
    return (q / (q.norm(dim=-1, keepdim=True) + _EPS)).view(-1, 84)


# =====================================================================
# 공개 API
# =====================================================================
def lowpass_window(quats, mode=None, window=None, edge=None, collect_stats=True):
    """윈도우 하나의 쿼터니언 시퀀스를 시간축으로 평활한다.

    quats : [F, 84] 모델이 출력한 21관절 로컬 쿼터니언.
            **hips 는 인자로 받지 않는다** (계획서 D5: 교정 대상이 아닌 통과값이라
            필터 대상이 아니다. 건드리면 87차원 계약과 MPJPE/intent 해석이 함께 흔들린다).

    반환: (quats_filtered [F, 84], stats dict)
      stats = {
        "mode", "window", "frames_total",
        "delta_deg_mean" : 입력 대비 평균 회전 변화량 [deg] (collect_stats=True 일 때만),
        "ms"             : 소요 시간 [ms],
      }

    보장:
      * 출력은 항상 관절별 단위 쿼터니언이다 (계획서 D3).
      * 부호 정렬 -> 필터 -> **부호 복원** 순서라, 필터가 항등이면(예: window=1,
        또는 시간축으로 상수인 입력) 출력이 입력과 수치적으로 같다.
        회전으로서는 q 와 -q 가 동치지만, 굳이 원래 부호로 되돌려 놓아 하위 코드가
        '모델 출력과 같은 반구'를 보게 한다.
      * 뼈 길이는 정의상 불변이다 (쿼터니언만 다루고 오프셋은 건드리지 않는다).
      * 난수를 쓰지 않으므로 결정론적이다.
    """
    mode = LP_MODE if mode is None else mode
    window = LP_WINDOW if window is None else window
    edge = LP_EDGE if edge is None else edge

    t0 = time.perf_counter()
    f = quats.shape[0]

    q3 = quats.view(f, 21, 4)
    q_aligned, flips = _hemispherize(q3)                      # D2 — 이걸 빼면 포즈가 깨진다
    x = q_aligned.reshape(f, 84)

    if mode == "one_euro":
        y = _one_euro(x, FPS, LP_ONE_EURO_MIN_CUTOFF, LP_ONE_EURO_BETA, LP_ONE_EURO_D_CUTOFF)
        window_eff = 0                                        # 인과 IIR — 창 개념이 없다
    else:
        window_eff = int(window)
        if window_eff % 2 == 0:
            raise ValueError(f"LP_WINDOW 는 홀수여야 한다: {window}")
        # 프레임 수가 창보다 짧으면 창을 줄인다 (30프레임 윈도우에서는 발생하지 않지만,
        # 짧은 시퀀스로 부르는 단위 검증·시각화 경로를 위해 방어한다).
        window_eff = min(window_eff, max(1, 2 * (f - 1) + 1))
        if window_eff <= 1:
            y = x
        else:
            if mode == "ma_zerophase":
                coeffs = _ma_coeffs(window_eff, x.dtype, x.device)
            elif mode == "savgol":
                coeffs = _savgol_coeffs(window_eff, LP_SAVGOL_POLY, x.dtype, x.device)
            else:
                raise ValueError(f"알 수 없는 LP_MODE: {mode}")
            y = _fir_zerophase(x, coeffs, edge)

    y = _normalize_quats(y)                                   # D3
    y = (y.view(f, 21, 4) * flips.unsqueeze(-1)).reshape(f, 84)   # 부호 복원

    stats = {
        "mode": mode,
        "window": window_eff,
        "frames_total": int(f),
        "delta_deg_mean": 0.0,
        "ms": 0.0,
    }
    if collect_stats:
        with torch.no_grad():
            dot = (y.view(f, 21, 4) * q3).sum(dim=-1).abs().clamp(max=1.0)
            stats["delta_deg_mean"] = float(2.0 * torch.rad2deg(torch.acos(dot)).mean())
    stats["ms"] = (time.perf_counter() - t0) * 1000.0
    return y, stats
