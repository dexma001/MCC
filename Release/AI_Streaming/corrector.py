"""스트리밍 보정기 — 프레임이 하나씩 들어오고 하나씩 나가는 형태로 감싼 배포 런타임.

[구조]
    push(frame) -> (out_frame | None, info)

  내부는 30프레임 링버퍼다. 버퍼가 차면 윈도우를 만들어
        모델 -> 저역통과(lowpass) -> 사영(projection)
  순서로 처리하고, 그 윈도우의 **끝쪽 stride 프레임**을 출력 큐에 넣는다.
  이 순서는 취향이 아니라 강제다 — 필터를 사영 뒤에 두면 저역통과가 사영 결과를 뭉개
  관통이 되살아난다 (실측: persistent max_pen4 0.138 -> 4.939cm).

[배포 갭 보고서(2026-09-11)에서 지적된 것 중 여기서 해결한 것]
  P0-2 inference_mode 크래시 : 진입 시 검사해 **명확한 메시지로 즉시 중단**한다.
       (사영은 내부에서 autograd.grad 를 쓴다. inference_mode 텐서로는 불가능하고,
        클린 입력에서는 위반 프레임이 0이라 그 경로를 타지 않아 **스모크가 통과해 버린다**.)
  P0-3 30프레임 미만        : 버퍼가 찰 때까지 **원본 통과**(warmup bypass). 조용히 다른
       답을 내는 대신 info["stage"]="warmup" 으로 드러낸다.
  P1-2 지연 상한 없음        : 남은 프레임 예산으로 사영 반복 수 K 를 **깎는다**(adaptive_k).
       사영 1회 반복 비용을 EMA 로 추정해 예산 안에서만 돈다. 깎인 횟수는 통계에 남는다.
  P2-1 계약 검증 없음        : validate_every>0 이면 주기적으로 입력 프레임을 검사한다.
  P3-1 패닉 스위치 없음      : corrector.bypass = True 로 언제든 원본 통과로 되돌린다.

[여기서 해결하지 않은 것 — 알고 남긴다]
  * do-no-harm (P1-1) 은 재학습 문제다. 런타임에서 고칠 수 없다.
  * 4쌍 밖 전신 관통은 여전히 보장 밖이다.
  * 30fps 가 아닌 입력의 리샘플링은 수신부(sources/OSC 어댑터)의 몫이다.
"""

import time
from collections import deque

import paths  # noqa: F401

import torch

import contract
import lowpass
import projection

STATS_WINDOW = 9000
"""분위수(p50/p95)를 계산할 최근 프레임 수 (9000 = 30fps 5분).

예전에는 프레임마다 리스트에 append 하고 summary() 가 전체를 정렬했다. 상시 실행에서는
메모리가 시간에 비례해 늘고(≈14MB/h) summary() 가 브리지 메인 루프를 멈춘다
(실측: 1h 분량 29ms, 8h 352ms, 24h 1.4s — 브리지는 5초마다 호출한다).
⇒ 분위수는 최근 창에서만, 개수·합·최대·예산 초과 수는 누적값으로 정확히 유지한다.
   창보다 짧은 실행(run_sim 600프레임 등)에서는 예전과 같은 숫자가 나온다."""


class _Series:
    """프레임별 수치: 최근 window 개(분위수용) + 전 구간 누적(개수·합·최대·임계 초과 수)."""

    def __init__(self, window=STATS_WINDOW, over=None):
        self.recent = deque(maxlen=window)
        self.over = over                 # 이 값을 넘은 개수를 셀 임계 (None = 세지 않음)
        self.count = 0
        self.total = 0.0
        self.max = None
        self.n_over = 0

    def append(self, x):
        self.recent.append(x)
        self.count += 1
        self.total += x
        if self.max is None or x > self.max:
            self.max = x
        if self.over is not None and x > self.over:
            self.n_over += 1

    def mean(self):
        return self.total / max(1, self.count)


class StreamingCorrector:
    """한 번에 한 프레임을 받아 한 프레임을 돌려주는 보정기 (상태를 가진 객체)."""

    def __init__(self, ckpt_path, device="cpu", model_class=None,
                 seq_len=contract.SEQ_LEN, stride=1, emit_lag=0,
                 budget_ms=1000.0 / contract.FPS,
                 lp_enabled=True, proj_enabled=True, proj_k=None,
                 adaptive_k=True, validate_every=0, physics=None):
        """
        ckpt_path    : 체크포인트 .pth 절대 경로
        model_class  : 기본은 AI_model/models.py 의 TransformerDenoiserCompat.
                       확장 아키텍처를 돌리려면 그 Compat 클래스를 넘긴다.
        stride       : 몇 프레임마다 한 번 계산할지. 1 = 완전 슬라이딩(품질 최고, 비용 최대),
                       seq_len = 타일링(경계 팝 +4.90cm/frame 실측).
        emit_lag     : 윈도우 끝에서 몇 프레임 뒤를 내보낼지. 0 = 최신 프레임(추가 지연 0).
                       2 로 두면 영위상 필터(창 5)가 양쪽 이웃을 다 가진 프레임을 내보낸다
                       (대가: 2프레임 ≈ 67ms 지연).
                       ⇒ 출력 지연 self.delay = (stride - 1) + emit_lag 프레임, **워밍업부터 일정**.
                          스트림 첫 delay 번의 push 는 None 을 돌려준다.
        budget_ms    : 프레임당 시간 예산. adaptive_k 가 이 값을 기준으로 K 를 깎는다.
        proj_k       : 사영 반복 상한. None 이면 projection.PROJ_K.
        validate_every: N>0 이면 N 프레임마다 입력 계약을 검사한다 (0 = 끄기).
        """
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "StreamingCorrector 는 torch.inference_mode() 안에서 만들 수 없습니다.\n"
                "  사영 레이어가 내부에서 autograd.grad 를 사용하므로 inference_mode 텐서로는\n"
                "  동작하지 않습니다. torch.no_grad() 를 쓰거나 grad 모드를 끄지 마세요.\n"
                "  (클린 입력에서는 이 경로를 타지 않아 스모크 테스트가 통과합니다 — 방송 중\n"
                "   첫 자기충돌에서 처음 죽습니다.)")

        self.device = torch.device(device)
        self.seq_len = int(seq_len)
        self.stride = max(1, int(stride))
        self.emit_lag = max(0, int(emit_lag))
        if self.stride + self.emit_lag > self.seq_len:
            raise ValueError(f"stride({self.stride}) + emit_lag({self.emit_lag}) 는 "
                             f"seq_len({self.seq_len}) 이하여야 합니다.")
        self.delay = self.stride - 1 + self.emit_lag
        """출력 지연(프레임). push 번호 t 에서는 항상 입력 프레임 t - delay 를 내보낸다."""
        self.budget_ms = float(budget_ms)
        self.lp_enabled = bool(lp_enabled)
        self.proj_enabled = bool(proj_enabled)
        self.proj_k = projection.PROJ_K if proj_k is None else int(proj_k)
        self.adaptive_k = bool(adaptive_k)
        self.validate_every = int(validate_every)
        self.bypass = False          # 패닉 스위치 — 언제든 True 로 두면 원본 통과

        if model_class is None:
            from models import TransformerDenoiserCompat as model_class
        self.model = model_class(input_dim=87, output_dim=84, latent_dim=64).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()
        self.ckpt_path = ckpt_path

        self.physics = physics if physics is not None else contract.make_physics(device)
        self.pairs = contract.COLLIDING_PAIRS

        self._buf = deque(maxlen=self.seq_len)      # 입력 링버퍼 [87] 프레임들
        self._out_q = deque()                       # 출력 대기열 (절대 번호, 프레임, 출처) 시간 순
        #   출처(stage)를 함께 들고 다닌다 — 출력에 남은 관통이 "보정 실패"인지
        #   "워밍업 구간의 원본 통과"인지 구분하지 못하면 판정이 통째로 틀린다.
        #   절대 번호를 들고 다니는 이유: 번호 없이 큐에 쌓기만 하면 stride>1 / emit_lag>0 에서
        #   워밍업 때 이미 내보낸 프레임을 첫 계산이 다시 넣어 **출력 시간이 뒤로 점프**했다
        #   (stride=30: 26,27,28 → 0,1,2…, 이후 29프레임 지연. 2026-10-03 수정).
        self._t = -1                                # 이번 세션에서 마지막으로 push 된 프레임 번호
        self._since_compute = 0
        self._n_push = 0
        # 사영 비용은 '고정 검출비 + 반복비 × K' 로 분해해야 한다.
        #   분해하지 않고 ms/iters 로 나누면 위반이 거의 없는 프레임(반복 0~1회)에서
        #   고정비가 통째로 반복비로 잡혀 **비용을 크게 과대평가**하고, 그 결과 K 가
        #   불필요하게 1까지 깎인다 (첫 구현에서 clean 600프레임 중 495회가 그랬다).
        self._proj_overhead_ms = 0.0                # 위반 0일 때의 검출 비용 EMA
        self._proj_ms_per_iter = 0.0                # 반복 1회당 추가 비용 EMA
        self._warmed = False                        # 콜드 스타트 1회 워밍업 여부

        self.stats = {
            "frames_in": 0, "frames_out": 0, "warmup_frames": 0, "bypass_frames": 0,
            "passthrough_frames": 0,
            "computes": 0, "k_capped": 0, "contract_problems": 0,
            "ms_total": _Series(over=self.budget_ms), "ms_model": _Series(), "ms_lp": _Series(),
            "ms_proj": _Series(),
            "proj_frames_touched": 0, "proj_residual_max_cm": 0.0, "proj_iters": _Series(),
        }

        # 콜드 스타트 비용은 **스트림이 시작되기 전에** 치른다.
        # 중립 포즈(단위 쿼터니언)로 한 번 돌려 모델·필터·사영(그래디언트 포함)을 데운다.
        # 측정: 이 호출이 137ms 를 먹는다. 스트림 루프 안에서 치르면 그 프레임이 통째로 밀린다.
        neutral = torch.zeros(contract.FRAME_DIM, dtype=torch.float32, device=self.device)
        neutral[3 + 3::4] = 1.0                     # 관절별 (x,y,z,w) 의 w = 1
        self._warm_once(neutral)

    # ------------------------------------------------------------------
    def reset(self):
        """세션 리셋 — 링버퍼·출력 큐를 비운다 (트래커가 끊겼다 다시 붙을 때). 모델·통계는 유지."""
        self._buf.clear()
        self._out_q.clear()
        self._t = -1
        self._since_compute = 0

    # ------------------------------------------------------------------
    def push(self, frame87):
        """프레임 하나를 밀어 넣고, 내보낼 프레임 하나를 돌려준다.

        반환: (out_frame [87] 또는 None, info dict)
          out_frame 은 항상 입력 프레임 (이번 번호 - self.delay) 에 해당한다.
          None 은 '아직 내보낼 것이 없다'는 뜻이다 (delay>0 일 때 스트림 첫 delay 번).
          info["emitted_stage"] = 내보낸 프레임의 출처:
            corrected   — 모델→필터→사영을 거친 프레임
            warmup      — 버퍼가 차기 전의 원본 통과
            bypass      — 패닉 스위치로 원본 통과
            passthrough — 그 밖에 보정본이 없어 원본을 통과시킨 경우
                          (stride>1 에서 bypass 를 풀거나 한 직후의 틈. 기본 설정에서는 생기지 않는다)
        """
        t0 = time.perf_counter()
        f = torch.as_tensor(frame87, dtype=torch.float32).reshape(contract.FRAME_DIM).to(self.device)
        self._n_push += 1
        self._t += 1
        self.stats["frames_in"] += 1
        info = {"stage": None, "ms": 0.0, "k_used": 0, "k_capped": False}

        if self.validate_every and self._n_push % self.validate_every == 1:
            problems = contract.validate_frame(f, physics=self.physics)
            if problems:
                self.stats["contract_problems"] += 1
                info["contract_problems"] = problems

        self._buf.append(f)

        if self.bypass:
            self._out_q.clear()                  # 패닉 스위치는 대기 중인 보정본도 즉시 버린다
            info["stage"] = "bypass"
            raw_stage = "bypass"
        elif len(self._buf) < self.seq_len:
            # P0-3: 버퍼가 찰 때까지 원본 통과. 29프레임 이하를 모델에 넣으면 조용히 다른 답이 나온다.
            info["stage"] = "warmup"
            raw_stage = "warmup"
        else:
            if self._since_compute == 0:
                self._compute_window(info)
                self.stats["computes"] += 1
            else:
                info["stage"] = "reuse"
            self._since_compute = (self._since_compute + 1) % self.stride
            raw_stage = "passthrough"

        out, emitted_stage = self._emit(raw_stage)
        info["emitted_stage"] = emitted_stage
        if emitted_stage == "warmup":
            self.stats["warmup_frames"] += 1
        elif emitted_stage == "bypass":
            self.stats["bypass_frames"] += 1
        elif emitted_stage == "passthrough":
            self.stats["passthrough_frames"] += 1
        if out is not None:
            self.stats["frames_out"] += 1
        info["ms"] = (time.perf_counter() - t0) * 1000.0
        self.stats["ms_total"].append(info["ms"])
        info["queue"] = len(self._out_q)
        return out, info

    # ------------------------------------------------------------------
    def _emit(self, raw_stage):
        """이번 push 에서 내보낼 프레임 = 입력 번호 (self._t - self.delay).

        대기열에 그 번호의 보정본이 있으면 그것을, 없으면 링버퍼의 원본을 raw_stage 로 내보낸다.
        번호가 지난 대기열 항목은 버린다 — 출력 시간축이 절대 뒤로 가지 않게 하는 유일한 지점이다.
        """
        target = self._t - self.delay
        if target < 0:
            return None, None
        while self._out_q and self._out_q[0][0] < target:
            self._out_q.popleft()
        if self._out_q and self._out_q[0][0] == target:
            _, frame, stage = self._out_q.popleft()
            return frame, stage
        # 링버퍼의 맨 앞 번호 = self._t - len(buf) + 1. delay <= seq_len-1 이라 target 은 늘 버퍼 안이다.
        pos = target - (self._t - len(self._buf) + 1)
        return self._buf[pos].clone(), raw_stage

    # ------------------------------------------------------------------
    def _warm_once(self, frame):
        """첫 계산의 '콜드 스타트'를 워밍업 구간에서 미리 치른다. 결과는 버린다.

        측정으로 드러난 문제: '첫 실제 위반 프레임'의 사영이 30ms 를 먹어 그 프레임만 예산을
        넘겼다 (p95 6.9ms 인데 max 35ms). autograd 그래프·커널이 처음 만들어지는 비용이다.
        생성자에서 호출하므로 이 비용은 스트림이 시작되기 전에 끝난다.

        [핵심] 사영의 **그래디언트 경로까지** 데워야 의미가 있다. 두 margin 을 모두 키워야
        하며(아래 주석), 위반 프레임 수에 따라 마스크 텐서 모양이 달라 여러 크기로 돌린다.
        실측 효과: 첫 위반 프레임의 사영 30ms -> 13ms.
        """
        self._warmed = True
        try:
            win = frame.unsqueeze(0).repeat(self.seq_len, 1)      # 같은 포즈 30프레임
            with torch.no_grad():
                out = self.model(win.unsqueeze(0))
                out = out[0] if isinstance(out, tuple) else out
                q = out[0, :, 3:].contiguous()
            if self.lp_enabled:
                q, _ = lowpass.lowpass_window(q, collect_stats=False)
            if self.proj_enabled:
                # [함정] detect_margin 만 키우면 프레임이 '위반'으로 잡히기는 해도
                # 교정 목표(margin_cm)로 잰 깊이가 0 이라 루프가 즉시 끝나 **autograd 경로에
                # 진입하지 않는다**. 실제로 그렇게 짰다가 워밍업이 무효였다(스파이크 그대로).
                # 두 margin 을 모두 키워야 그래디언트 계산까지 데워진다.
                # 위반 프레임 수마다 마스크 텐서 모양이 다르므로 1~3개와 전체를 함께 만든다.
                for n in (1, 2, 3, self.seq_len):
                    projection.project_window(self.physics, win[:n, :3], q[:n], self.pairs,
                                              k=2, margin_cm=100.0, detect_margin_cm=100.0,
                                              collect_stats=False)
        except Exception:
            pass          # 워밍업 실패는 치명적이지 않다 — 첫 프레임이 느려질 뿐이다

    # ------------------------------------------------------------------
    def _compute_window(self, info):
        """버퍼가 가득 찬 상태에서 한 윈도우를 처리해 출력 큐에 stride 프레임을 넣는다."""
        window = torch.stack(list(self._buf), dim=0)            # [S, 87]
        hips = window[:, :3]
        info["stage"] = "compute"

        # --- 1. 모델 ---------------------------------------------------
        t = time.perf_counter()
        with torch.no_grad():
            out = self.model(window.unsqueeze(0))
            out = out[0] if isinstance(out, tuple) else out     # Compat 는 3-튜플
            quats = out[0, :, 3:].contiguous()                  # [S, 84]
        ms_model = (time.perf_counter() - t) * 1000.0

        # --- 2. 저역통과 (반드시 사영 앞) --------------------------------
        ms_lp = 0.0
        if self.lp_enabled and lowpass.LP_ENABLED:
            t = time.perf_counter()
            quats, _ = lowpass.lowpass_window(quats, collect_stats=False)
            ms_lp = (time.perf_counter() - t) * 1000.0

        # --- 3. 사영 (예산에 맞춰 K 를 깎는다) ---------------------------
        ms_proj = 0.0
        k_used = 0
        if self.proj_enabled and projection.PROJ_ENABLED:
            k = self.proj_k
            if self.adaptive_k and self._proj_ms_per_iter > 0:
                remaining = (self.budget_ms * self.stride - (ms_model + ms_lp)
                             - self._proj_overhead_ms)
                k_allowed = int(remaining / self._proj_ms_per_iter)
                if k_allowed < k:
                    k = max(1, k_allowed)
                    info["k_capped"] = True
                    self.stats["k_capped"] += 1
            t = time.perf_counter()
            quats, pstats = projection.project_window(
                self.physics, hips, quats, self.pairs, k=k, collect_stats=False)
            ms_proj = (time.perf_counter() - t) * 1000.0
            k_used = int(pstats.get("iters_used", 0))
            if k_used == 0:
                # 위반 0 = 검출만 하고 끝난 경우. 이때의 시간이 곧 고정 검출비다.
                self._proj_overhead_ms = (ms_proj if self._proj_overhead_ms == 0
                                          else 0.8 * self._proj_overhead_ms + 0.2 * ms_proj)
            elif k_used >= 2:
                # 반복비는 '반복이 실제로 여러 번 돈' 경우에서만 뽑는다 (고정비 오염 방지).
                sample = max(0.0, ms_proj - self._proj_overhead_ms) / k_used
                self._proj_ms_per_iter = (sample if self._proj_ms_per_iter == 0
                                          else 0.8 * self._proj_ms_per_iter + 0.2 * sample)
            self.stats["proj_frames_touched"] += int(pstats.get("frames_touched", 0))
            self.stats["proj_residual_max_cm"] = max(
                self.stats["proj_residual_max_cm"], float(pstats.get("residual_max_cm", 0.0)))
            self.stats["proj_iters"].append(k_used)
            info["proj_frames_touched"] = int(pstats.get("frames_touched", 0))
            info["proj_residual_max_cm"] = float(pstats.get("residual_max_cm", 0.0))

        info["k_used"] = k_used
        self.stats["ms_model"].append(ms_model)
        self.stats["ms_lp"].append(ms_lp)
        self.stats["ms_proj"].append(ms_proj)

        # --- 4. 출력 큐에 stride 프레임 --------------------------------
        corrected = torch.cat([hips, quats], dim=-1)            # [S, 87]
        end = self.seq_len - self.emit_lag                      # 내보낼 마지막 인덱스(배타)
        start = end - self.stride
        first_abs = self._t - self.seq_len + 1                  # 윈도우 0번 프레임의 절대 번호
        for i in range(start, end):
            self._out_q.append((first_abs + i, corrected[i].clone(), "corrected"))

    # ------------------------------------------------------------------
    def summary(self):
        """지연·개입률 요약. 라이브 판정에 쓰는 숫자는 평균이 아니라 p95/max 다."""
        def pct(xs, p):
            if not xs:
                return 0.0
            s = sorted(xs)
            return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]

        total = self.stats["ms_total"]
        comp = [t for t in total.recent if t > 0]          # 분위수는 최근 STATS_WINDOW 프레임
        recent_over = sum(1 for t in total.recent if t > self.budget_ms)
        st = self.stats
        return {
            "frames_in": self.stats["frames_in"],
            "frames_out": self.stats["frames_out"],
            "warmup_frames": self.stats["warmup_frames"],
            "bypass_frames": self.stats["bypass_frames"],
            "passthrough_frames": self.stats["passthrough_frames"],
            "delay_frames": self.delay,
            "computes": self.stats["computes"],
            "budget_ms": self.budget_ms,
            "ms_p50": pct(comp, 50), "ms_p95": pct(comp, 95),
            "ms_max": total.max or 0.0,                       # 이하 누적값 (전 구간, 정확)
            "over_budget_frames": total.n_over,
            "over_budget_pct": 100.0 * total.n_over / max(1, total.count),
            "ms_model_mean": st["ms_model"].mean(),
            "ms_lp_mean": st["ms_lp"].mean(),
            "ms_proj_mean": st["ms_proj"].mean(),
            "ms_proj_max": st["ms_proj"].max or 0.0,
            "k_capped": self.stats["k_capped"],
            "proj_iters_max": st["proj_iters"].max or 0,
            # 최근 창 (상시 실행에서 '지금' 상태를 보는 값 — 누적 %는 오래 돌면 최근 스파이크를 묻는다)
            "window_frames": len(total.recent),
            "recent_ms_max": max(total.recent) if total.recent else 0.0,
            "recent_over_budget_pct": 100.0 * recent_over / max(1, len(total.recent)),
            "proj_frames_touched": self.stats["proj_frames_touched"],
            "proj_residual_max_cm": self.stats["proj_residual_max_cm"],
            "contract_problems": self.stats["contract_problems"],
        }
