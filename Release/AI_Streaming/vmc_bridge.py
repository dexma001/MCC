"""VMC 브리지 — 트래커(VMC 송신) → 보정 → Warudo(VMC 수신).  실제 송수신 진입점.

    트래커 ──VMC(UDP)──▶ [이 프로세스: 수신 → VRM→리그 → 보정 → 리그→VRM → 송신] ──VMC──▶ Warudo
                            (39540 등)                                              (39539)

실행 예:
    VSCode "Run Python File" — 아래 [실행 설정] 블록(MODE 등)을 바꾸고 인자 없이 실행 (train.py 와 같은 규칙)
    python AI_Streaming/vmc_bridge.py --loopback-test                       # 소켓·관통 제거 자체 검사 (Warudo·데이터셋 불필요)
    python AI_Streaming/vmc_bridge.py --replay clean --send-port 39539       # 데이터셋 모션을 Warudo 로 재생
    python AI_Streaming/vmc_bridge.py --listen-port 39540 --send-port 39539  # 라이브 (트래커 → Warudo)
    python AI_Streaming/vmc_bridge.py --listen-port 39540 --dry-run          # 수신·보정만, 전송 안 함
    python AI_Streaming/tracker_sim.py --scenario persistent                 # (다른 창) 가상 트래커로 실측

[Warudo 쪽 설정]
  * Warudo: 캐릭터 → 모션 캡처 → "VMC" 수신기, 포트 39539 (Warudo 기본값). VRM 모델 사용.
  * 트래커(VMC 앱/VSeeFace/mocopi 등)의 송신 대상을 **이 브리지의 --listen-port** 로 바꾼다.
    트래커가 Warudo 로 직접 보내면서 브리지도 보내면 두 스트림이 겹쳐 아바타가 떤다.
  * Warudo 의 "VMC Sender" 에셋을 입력으로 쓸 수도 있다. 단, 같은 캐릭터가 VMC 수신도 하면
    송신 값이 보정된 값이라 **되먹임 고리**가 된다 — 입력 캐릭터와 출력 캐릭터를 분리하거나
    외부 트래커를 쓴다.

[이 파일이 하는 일 — 순서대로]
  1. 수신: /VMC/Ext/Bone/Pos 중 우리 21본만 프레임으로 모은다. 그 밖의 본(손가락·눈·UpperChest),
     /VMC/Ext/Blend/*, Root/Pos 는 **그대로 통과**시킬 목록에 넣는다(모델 대상이 아니다).
     /VMC/Ext/T 를 프레임 경계로 센다. /VMC/Ext/OK 는 버린다(우리 것을 보낸다).
  2. UpperChest 접기: 트래커가 UpperChest 를 보내면 Chest := Chest·UpperChest 로 합쳐 모델에 넣고
     (Unity 에서 Neck/어깨의 부모가 UpperChest 이므로 이렇게 해야 월드 방향이 같다),
     보낼 때는 Chest_out·inv(UpperChest) 와 UpperChest(원본)로 되돌린다.
  3. 세션 정규화(학습 데이터의 "Normalized" 와 같은 규약): 첫 프레임의 Hips 위치를 원점으로,
     첫 프레임의 앞 방향(다리 위치로 측정)을 +Z 로 돌린다. 모델 입력만 바꾸고 출력에서 되돌린다.
  4. VRM→리그 변환(retarget) → StreamingCorrector.push → 리그→VRM 변환.
  5. 30fps 시계로 돈다. 트래커가 더 빠르면 최신 프레임만 쓰고(sample-and-hold), 느려서 새 프레임이
     없는 틱(hold)은 **모델에 넣지도 보내지도 않는다**(2026-10-07 — 같은 프레임 재주입이 창에 머무는
     1초 동안 출력을 흔들었다). 트래커 수신이 0.5초 없으면 송신을 멈추고(Warudo 는 마지막 포즈 유지),
     2초 이상 끊겼다 돌아오면 세션을 리셋한다.
  5b. 출력은 윈도우 끝에서 EMIT_LAG(기본 2) 프레임 앞의 것을 낸다(+67ms). 끝자리는 미래 문맥이 없어
     윈도우마다 답이 달라 튄다(claude_analysis/streaming_jitter_report_20261006.md).
  6. 보정에서 예외가 나면 그 프레임은 원본을 보낸다(방송이 끊기지 않는다). 키 'b' 로 bypass 토글, 'q' 종료.

[여기서 해결하지 않은 것 — 알고 남긴다]
  * 트래커의 fps 가 30 이 아니면 sample-and-hold 로만 맞춘다(보간 없음). 60fps 트래커면 절반은 버린다.
  * 아바타의 발 기울기·손바닥 방향은 리그 T-포즈(rig_tpose.json)의 가정과 다를 수 있다 — 위치는 맞고
    본의 롤만 어긋난다. 눈으로 보고 판단할 항목.
  * 지연 = 트래커 주기(≤33ms) + 보정(5~38ms) + Warudo 렌더. persistent 상황의 상한은 여전히 없다.
"""

import argparse
import math
import os
import sys
import threading
import time
from collections import deque

import paths

import torch

import contract
import retarget as rt
import sinks
from corrector import StreamingCorrector

BODY_BONES = frozenset(contract.BONE_ORDER)

# =====================================================================
# [실행 설정] VSCode "Run Python File" 용 — train.py 와 같은 규칙: 여기 값을 바꾸고 실행한다.
#   인자 없이 실행했을 때(= Run Python File, F5)만 이 값들을 쓴다.
#   명령줄 인자를 하나라도 주면(server.py 가 띄울 때 포함) 이 블록은 무시되고 argparse 기본값이 쓰인다
#   — 여기를 바꿔도 server.py·README 의 명령어 동작은 변하지 않는다.
#   실행 시 이 설정과 같은 명령줄을 출력하므로 터미널에서 재현할 때 그대로 복사하면 된다.
# =====================================================================
MODE = "live"               # "live" = 트래커 수신 → Warudo  |  "replay" = 데이터셋 재생 → Warudo
                            # "loopback" = Warudo·데이터셋 없이 소켓 왕복 + 관통 제거 자체 검사 (끝나면 종료)
REPLAY_SCENARIO = "persistent"   # MODE="replay" 일 때: "clean" | "transient" | "persistent" | "legacy80"
REPLAY_HIPS_HEIGHT = 0.9    # MODE="replay" 일 때 Hips 절대 높이(m). 아바타가 뜨거나 잠기면 조정
REPLAY_LEVEL_GROUND = True  # MODE="replay" 일 때 데이터셋의 경사 바닥을 평지로 편다 (sources.level_ground)
                            #   False = 원본 그대로 — 걷기·달리기에서 아바타가 비스듬히 올라간다

LISTEN_PORT = 39540         # 트래커가 보내는 포트 (트래커의 송신 대상을 이 값으로)
SEND_HOST = "127.0.0.1"     # Warudo 가 도는 PC
SEND_PORT = 39539           # Warudo VMC 수신 포트 (Warudo 기본값)
DRY_RUN = False             # True = 수신·보정만, 전송 안 함

LOWPASS = True              # False = 저역통과 끔
PROJECTION = True           # False = 사영 끔 (관통 보장 사라짐)
ADAPTIVE_K = True
FOLD_UPPERCHEST = True
LOG_CSV = None              # 예: "bridge_log.csv" — 프레임별 실측 기록 (실행 위치 기준 상대 경로)
STATUS_PORT = 0             # 상태·제어 HTTP 포트 (0 = 끔)
THREADS = 0                 # torch CPU 스레드 수 (0 = torch 기본값)
EMIT_LAG = 2                # 윈도우 끝에서 몇 프레임 앞을 내보낼지. 0 = 끝자리(지연 최소, 튐 큼)
                            #   2 = +67ms 지연으로 튐 완화 (2026-10-07 기본값)
HOLD_REPUSH = False         # True = 새 트래커 프레임이 없는 틱에도 같은 프레임을 모델에 다시 넣는다
                            #   (2026-10-07 이전 동작 — 비교·롤백용)


def _argv_from_settings():
    """위 [실행 설정] 블록 → 명령줄 인자 목록. 검증은 argparse 가 그대로 한다."""
    if MODE not in ("live", "replay", "loopback"):
        raise ValueError(f'MODE 는 "live" | "replay" | "loopback" 중 하나여야 합니다: {MODE!r}')
    argv = []                   # 가중치는 늘 checkpoints/temp.pth (고를 것이 없다)
    if MODE == "loopback":
        return argv + ["--loopback-test"]
    if MODE == "replay":
        argv += ["--replay", REPLAY_SCENARIO, "--replay-hips-height", str(REPLAY_HIPS_HEIGHT)]
        if not REPLAY_LEVEL_GROUND:
            argv.append("--replay-no-level")
    else:
        argv += ["--listen-port", str(LISTEN_PORT)]
    argv += ["--send-host", SEND_HOST, "--send-port", str(SEND_PORT)]
    if DRY_RUN:
        argv.append("--dry-run")
    if not LOWPASS:
        argv.append("--no-lowpass")
    if not PROJECTION:
        argv.append("--no-projection")
    if not ADAPTIVE_K:
        argv.append("--no-adaptive-k")
    if not FOLD_UPPERCHEST:
        argv.append("--no-fold-upperchest")
    if LOG_CSV:
        argv += ["--log-csv", LOG_CSV]
    if STATUS_PORT:
        argv += ["--status-port", str(STATUS_PORT)]
    if THREADS:
        argv += ["--threads", str(THREADS)]
    argv += ["--emit-lag", str(EMIT_LAG)]
    if HOLD_REPUSH:
        argv.append("--hold-repush")
    return argv


# =====================================================================
# 수신
# =====================================================================
class VmcReceiver:
    """OSC 서버 스레드. 우리 21본의 최신 값과 pass-through 메시지를 보관한다."""

    def __init__(self, port, host="0.0.0.0"):
        from pythonosc import dispatcher, osc_server
        self.port = port
        self._lock = threading.Lock()
        self._bones = {}                 # name -> [px,py,pz,qx,qy,qz,qw]
        self._upperchest = None          # [qx,qy,qz,qw] (있을 때만)
        self._root = None
        self._pass = []                  # [(addr, args)] 마지막 take() 이후 누적
        self._dirty = False              # 마지막 take() 이후 본 갱신이 있었나
        self.n_frames = 0                # /VMC/Ext/T 로 센 트래커 프레임 수
        self.n_msgs = 0
        self.last_rx = 0.0
        d = dispatcher.Dispatcher()
        d.map(sinks.VmcOscSink.BONE_ADDR, self._on_bone)
        d.map(sinks.VmcOscSink.ROOT_ADDR, self._on_root)
        d.set_default_handler(self._on_other)
        self._server = osc_server.ThreadingOSCUDPServer((host, port), d)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._server.shutdown()
        self._server.server_close()

    # --- 핸들러 (서버 스레드) ---------------------------------------
    def _on_bone(self, addr, *args):
        self.n_msgs += 1
        self.last_rx = time.perf_counter()
        if len(args) != 8:
            return
        name = args[0]
        with self._lock:
            if name in BODY_BONES:
                self._bones[name] = [float(x) for x in args[1:]]
                self._dirty = True
            elif name == "UpperChest":
                self._upperchest = [float(x) for x in args[4:8]]
                self._pass.append((addr, list(args)))
            else:
                self._pass.append((addr, list(args)))          # 손가락·눈 등: 그대로 통과

    def _on_root(self, addr, *args):
        self.n_msgs += 1
        with self._lock:
            self._root = list(args)

    def _on_other(self, addr, *args):
        self.n_msgs += 1
        self.last_rx = time.perf_counter()
        if addr == sinks.VmcOscSink.T_ADDR:
            self.n_frames += 1
        elif addr == sinks.VmcOscSink.OK_ADDR:
            return
        else:
            with self._lock:
                self._pass.append((addr, list(args)))

    # --- 소비 (메인 스레드) -----------------------------------------
    def take(self):
        """(VRM 규약 [87] 또는 None, upperchest quat 또는 None, pass-through 목록, root, 갱신 여부)."""
        with self._lock:
            missing = [b for b in contract.BONE_ORDER if b not in self._bones]
            frame = None
            if not missing:
                hips = self._bones["Hips"][:3]
                frame = contract.pack_frame(hips, {b: self._bones[b][3:] for b in contract.BONE_ORDER})
            passthrough, self._pass = self._pass, []
            dirty, self._dirty = self._dirty, False
            return frame, self._upperchest, passthrough, self._root, dirty, missing


# =====================================================================
# 세션 정규화 — 학습 데이터의 "Normalized" 규약을 라이브에 재현
# =====================================================================
class SessionNormalizer:
    """첫 프레임 기준 Hips 원점화 + 앞 방향 +Z 정렬. 모델 입력에만 적용하고 출력에서 되돌린다.

    학습 CSV("VMC_Normalized")의 정규화를 실측한 결과: 각 파일에 **상수 월드 회전 R**(yaw, 일부는
    up-axis 보정)을 Hips 쿼터니언에 곱하고 Hips 위치를 R·(p − p0) 로 두었다(다른 본은 불변).
    라이브에서도 같은 규약이어야 모델이 학습 분포의 입력을 본다.
    """

    def __init__(self, retarget, physics):
        self.retarget = retarget
        self.physics = physics
        self.reset()

    def reset(self):
        self.p0 = None
        self.R = None            # 월드 회전 (x,y,z,w): 앞 → +Z
        self.R_inv = None

    def _calibrate(self, frame_rig):
        f = frame_rig
        gp = self.physics.forward_kinematics(f[:3].reshape(1, 3), f[3:].reshape(1, 84))
        left = (gp['LeftUpperLeg'] - gp['RightUpperLeg']).reshape(3)
        fwd = torch.cross(rt._Y, left, dim=-1) * torch.tensor([1.0, 0.0, 1.0])
        if fwd.norm() < 1e-6:
            fwd = rt._Z.clone()
        self.R = rt.q_swing(fwd, rt._Z)
        self.R_inv = rt.qconj(self.R)
        self.p0 = f[:3].clone()
        self.yaw_deg = math.degrees(2 * math.atan2(float(self.R[1]), float(self.R[3])))

    def to_model(self, frame_vrm):
        """VRM 규약 [87] -> 리그 규약·정규화된 [87] (모델 입력)."""
        f = self.retarget.vrm_to_rig(frame_vrm)
        if self.R is None:
            self._calibrate(f)
        hips_q = f[3 + 4 * contract.BONE_INDEX['Hips']: 3 + 4 * contract.BONE_INDEX['Hips'] + 4]
        f = f.clone()
        f[:3] = rt.qrot(self.R, f[:3] - self.p0)
        i = 3 + 4 * contract.BONE_INDEX['Hips']
        f[i:i + 4] = rt.qmul(self.R, hips_q)
        return f

    def from_model(self, frame_rig):
        """모델 출력(리그·정규화) [87] -> VRM 규약 [87] (절대 Hips 위치·원래 방향)."""
        f = frame_rig.clone()
        i = 3 + 4 * contract.BONE_INDEX['Hips']
        f[i:i + 4] = rt.qmul(self.R_inv, f[i:i + 4])
        f[:3] = rt.qrot(self.R_inv, f[:3]) + self.p0
        return self.retarget.rig_to_vrm(f)


# =====================================================================
# UpperChest 접기
# =====================================================================
def fold_upperchest(frame_vrm, uc_quat):
    """Chest := Chest·UpperChest (VRM 규약). uc_quat None 이면 그대로."""
    if uc_quat is None:
        return frame_vrm
    f = frame_vrm.clone()
    i = 3 + 4 * contract.BONE_INDEX['Chest']
    uc = torch.tensor(uc_quat, dtype=torch.float32)
    f[i:i + 4] = rt.qnorm(rt.qmul(f[i:i + 4], uc))
    return f


def unfold_upperchest(frame_vrm, uc_quat):
    """Chest_out := Chest_total·inv(UpperChest). UpperChest 자체는 pass-through 로 이미 실린다."""
    if uc_quat is None:
        return frame_vrm
    f = frame_vrm.clone()
    i = 3 + 4 * contract.BONE_INDEX['Chest']
    uc = torch.tensor(uc_quat, dtype=torch.float32)
    f[i:i + 4] = rt.qnorm(rt.qmul(f[i:i + 4], rt.qconj(uc)))
    return f


# =====================================================================
# 브리지 본체
# =====================================================================
class Bridge:
    def __init__(self, corrector, sink, retarget, physics, fold_uc=True, gap_reset_s=2.0):
        self.corrector = corrector
        self.sink = sink
        self.norm = SessionNormalizer(retarget, physics)
        self.fold_uc = fold_uc
        self.gap_reset_s = gap_reset_s
        self.errors = 0
        self.consecutive_errors = 0
        self.send_errors = 0             # 소켓 송신 실패 (네트워크 끊김 등) — 프로세스를 죽이지 않는다
        self.last_send_error = ""
        self.frames = 0
        self.sent = 0
        self.last_out_vrm = None
        self._last_rx_frame_t = None
        # 관통 실측용 (step 시간 밖에서 잰다): 이번 틱의 입력 / 보낸 출력. 둘 다 UpperChest 를 접은 상태 =
        #   모델이 보는 몸통 = Warudo 가 Chest·UpperChest 로 합쳐 그리는 몸통. 보낸 것이 없으면 last_out_fold=None.
        self.last_in_fold = None
        self.last_out_fold = None

    def reset_session(self):
        self.norm.reset()
        self.corrector.reset()
        self.last_out_vrm = None

    def step(self, frame_vrm, uc_quat=None, passthrough=(), root=None):
        """트래커 프레임 하나(VRM 규약) -> Warudo 로 전송. 반환: (보낸 VRM 프레임 또는 None, info)."""
        now = time.perf_counter()
        if self._last_rx_frame_t is not None and now - self._last_rx_frame_t > self.gap_reset_s:
            self.reset_session()
        self._last_rx_frame_t = now
        self.frames += 1

        info = {}
        self.last_in_fold = self.last_out_fold = None
        try:
            fin = fold_upperchest(frame_vrm, uc_quat) if self.fold_uc else frame_vrm
            self.last_in_fold = fin
            x = self.norm.to_model(fin)
            out, info = self.corrector.push(x)
            if out is None:
                return None, info
            y = self.norm.from_model(out)
            y_fold = y
            y = unfold_upperchest(y, uc_quat) if self.fold_uc else y
            self.consecutive_errors = 0
        except Exception as e:                       # 방송 중 예외 = 원본 통과, 죽지 않는다
            self.errors += 1
            self.consecutive_errors += 1
            info = {"stage": "error", "error": f"{type(e).__name__}: {e}"}
            y = frame_vrm
            y_fold = self.last_in_fold if self.last_in_fold is not None else frame_vrm
        # y 는 이미 VRM 규약이다(from_model 이 변환함) — 싱크의 리그->VRM 변환은 건너뛴다.
        #   (예전에는 sink.retarget 을 영구히 바꿔치기해서 --no-retarget 에서도 보고가 '변환 ON' 이었다.)
        try:
            self.sink.send(y, extra=passthrough, root=root, already_vrm=True)
        except OSError as e:
            # 보정 예외와 달리 송신 실패는 예전에 try 밖이라 브리지 전체가 죽었다.
            #   (Warudo 가 다른 PC 이고 네트워크가 잠깐 끊기면 sendto 가 WSAENETUNREACH 등을 던진다.)
            #   다음 프레임에서 다시 보내면 되므로 세고 넘어간다.
            self.send_errors += 1
            self.last_send_error = f"{type(e).__name__}: {e}"
            info["send_error"] = self.last_send_error
            return None, info
        self.sent += 1
        self.last_out_vrm = y
        self.last_out_fold = y_fold
        return y, info


class PenetrationMeter:
    """콘솔 통계용 4쌍 관통 실측 (cm) — 브리지가 받은 입력과 Warudo 로 보낸 출력을 같은 자로 잰다.

    2026-10-09: 셸에 보이던 관통 숫자는 tracker_sim 의 '보정 전 입력' 누적 max 뿐이라(8.62cm 고정)
    보정이 안 되는 것처럼 보였다. 브리지 출력의 관통은 어디에도 찍히지 않았다(--log-csv 의 사영 잔차뿐).
    여기서는 보낸 프레임 자체를 FK 로 다시 재므로 모델·필터·사영 어느 단계를 믿지 않아도 된다.
    출력은 출처별로 나눈다: corrected(보정본) / 원본 통과(warmup·bypass·passthrough·error) —
    워밍업 1초의 원본 관통이 '보정 실패'로 읽히지 않게.
    비용: 2프레임 FK ≈ 2.7ms/틱 (CPU 1스레드 실측) — step 처리 시간(ms·예산초과)에는 넣지 않는다.
    """

    def __init__(self, retarget, physics):
        self.retarget = retarget
        self.physics = physics
        self.window = self._empty()
        self.session = self._empty()

    @staticmethod
    def _empty():
        return {"in": 0.0, "out": 0.0, "raw": 0.0, "n_raw": 0}

    def _pen(self, frames_vrm):
        r = self.retarget.vrm_to_rig(torch.stack(frames_vrm))
        d = self.physics.get_penetration_depths_from_quats(r[:, :3], r[:, 3:], contract.COLLIDING_PAIRS)
        return (d.amax(-1) * 100.0).tolist()

    def update(self, frame_in, frame_out, emitted_stage):
        """이번 틱의 입력·출력(None 가능)을 재서 창·세션 max 에 더한다. 반환: (입력 cm, 출력 cm 또는 None)."""
        frames = [f for f in (frame_in, frame_out) if f is not None]
        if not frames:
            return None, None
        pens = self._pen(frames)
        p_in = pens[0] if frame_in is not None else None
        p_out = pens[-1] if frame_out is not None else None
        for acc in (self.window, self.session):
            if p_in is not None:
                acc["in"] = max(acc["in"], p_in)
            if p_out is not None:
                if emitted_stage == "corrected":
                    acc["out"] = max(acc["out"], p_out)
                else:
                    acc["raw"] = max(acc["raw"], p_out)
                    acc["n_raw"] += 1
        return p_in, p_out

    @staticmethod
    def format(acc):
        s = f"관통 4쌍 max 입력 {acc['in']:5.2f} → 출력 {acc['out']:5.2f} cm"
        if acc["n_raw"]:
            s += f" (원본통과 {acc['n_raw']}f max {acc['raw']:.2f})"
        return s

    def pop_window(self):
        w, self.window = self.window, self._empty()
        return w


# =====================================================================
# 입력원 어댑터: 데이터셋 재생을 '트래커가 보낸 VRM 프레임'처럼 만든다
# =====================================================================
class ReplayAsTracker:
    """sources.ReplaySource 의 리그 프레임을 VRM 규약 + 절대 Hips 위치로 바꿔 내놓는다."""

    def __init__(self, scenario, retarget, physics, hips_height=0.9, files=8, max_frames=0, level=True):
        import sources
        self.src = sources.ReplaySource(paths.test_motion_files(limit=files), scenario=scenario,
                                        pace=False, physics=physics, max_frames=max_frames, loop=True,
                                        level=level)
        self.retarget = retarget
        self.offset = torch.tensor([0.0, hips_height, 0.0])

    def __iter__(self):
        for inp, _clean in self.src:
            f = self.retarget.rig_to_vrm(inp)
            f = f.clone()
            f[:3] = f[:3] + self.offset
            yield f


# =====================================================================
# 자체 검사: 소켓 왕복 + 관통 제거 (Warudo·데이터셋 없이)
# =====================================================================
def synthetic_penetrating_motion(retarget, n=90, fps=30.0):
    """매 프레임 4쌍이 모두 관통하는 합성 모션 [n, 87] (리그 규약, Hips 는 첫 프레임 기준 상대값).

    VRM T-포즈(본 회전 전부 0, 정면 +Z, 왼팔 −X)에서 시작해
      양팔: 아래로 80° 내리고 팔꿈치를 120° 접어 아래팔이 몸통을 뚫고 서로 교차 (아래팔↔몸통 ×2, 아래팔↔아래팔)
      양다리: 4° 모아 정강이가 겹침 (정강이↔정강이)
    을 만들고 각도·척추 비틀기·고개·Hips 위치를 천천히 흔든다. 실측 관통 깊이 2.0~5.2cm (4쌍 전부, 90프레임 전부).
    데이터셋이 필요 없어 배포판(Release)에서도 돈다. 학습 분포 밖의 자세라 보정 '품질'이 아니라
    전송·변환·정규화와 **관통 제거(사영의 그래디언트 경로)** 를 검사하는 용도다 — 깨끗한 입력만 보내면
    사영에 진입조차 하지 않아 그 경로의 결함(예: inference_mode 크래시)을 통과시켜 버린다.
    첫 프레임은 정면(+Z)을 향한다(SessionNormalizer 가 되돌릴 기준).
    """
    I = contract.BONE_INDEX
    tpose_vrm = retarget.rig_to_vrm(retarget.rig_tpose_frame())

    def qa(axis, deg):
        return rt.q_axis_angle(torch.tensor(axis, dtype=torch.float32), math.radians(deg))

    frames = []
    for i in range(n):
        t = i / fps
        s1 = math.sin(2 * math.pi * 0.5 * t)
        s2 = math.sin(2 * math.pi * 0.8 * t + 1.0)
        s3 = math.sin(2 * math.pi * 0.3 * t)
        arm_down, arm_fwd = 80 + 5 * s1, -20 + 5 * s2
        elbow_z, elbow_y = 120 + 8 * s3, -20 + 5 * s1
        leg_in = 4 + 0.5 * s2
        v = tpose_vrm.clone()

        def setq(bone, q):
            v[3 + 4 * I[bone]:3 + 4 * I[bone] + 4] = q
        setq("LeftUpperArm", rt.qmul(qa([0, 1, 0], arm_fwd), qa([0, 0, 1], arm_down)))
        setq("RightUpperArm", rt.qmul(qa([0, 1, 0], -arm_fwd), qa([0, 0, 1], -arm_down)))
        setq("LeftLowerArm", rt.qmul(qa([0, 1, 0], elbow_y), qa([0, 0, 1], elbow_z)))
        setq("RightLowerArm", rt.qmul(qa([0, 1, 0], -elbow_y), qa([0, 0, 1], -elbow_z)))
        setq("LeftUpperLeg", qa([0, 0, 1], leg_in))
        setq("RightUpperLeg", qa([0, 0, 1], -leg_in))
        setq("Spine", qa([0, 1, 0], 6 * s3))
        setq("Head", qa([1, 0, 0], 8 * s2))
        v[:3] = torch.tensor([0.03 * s3, 0.01 * s1, 0.02 * s2])
        frames.append(retarget.vrm_to_rig(v))
    motion = torch.stack(frames)
    motion[:, :3] = motion[:, :3] - motion[0, :3].clone()
    return motion


def loopback_test(corrector, retarget, physics, port_in=39555, port_out=39556, n=90, yaw_deg=37.0,
                  min_input_pen_cm=1.0, max_residual_cm=0.05):
    """합성 트래커 → 브리지 → 로컬 수신기. 실제 UDP 소켓과 번들 파싱, 관통 제거까지 검사한다.

    입력은 synthetic_penetrating_motion (매 프레임 4쌍 관통) — 데이터셋이 필요 없다
      (2026-10-08 부터. 그 전에는 held-out 데이터셋 첫 파일을 썼다).
    yaw_deg: 합성 트래커 전체를 월드 Y축으로 이만큼 돌려 보낸다. 합성 모션 첫 프레임은 정면(+Z)을
      보고 있어서 0° 로는 SessionNormalizer 의 회전 경로가 검사되지 않는다(예전 T15 가 그랬다:
      세션 yaw 0.0°). 돌려 보내면 정규화가 그 회전을 되돌려 모델이 원래 합성 프레임을 봐야 한다.
    min_input_pen_cm: 입력 모든 프레임·4쌍의 관통이 이 이상이어야 한다(검사가 공허하지 않게).
    max_residual_cm: 수신측에서 잰 보정 프레임의 4쌍 관통 상한 (UDP float32 왕복 오차 여유).
    """
    from pythonosc.udp_client import SimpleUDPClient
    print("[loopback] 수신기 2개 기동:", port_in, port_out)
    rx_in = VmcReceiver(port_in, host="127.0.0.1").start()      # 브리지의 입력 (트래커 역할이 보낸다)
    rx_out = VmcReceiver(port_out, host="127.0.0.1").start()    # Warudo 역할 (브리지가 보낸다)
    tracker = sinks.VmcOscSink(client=SimpleUDPClient("127.0.0.1", port_in), retarget=retarget)
    sink = sinks.VmcOscSink(client=SimpleUDPClient("127.0.0.1", port_out), retarget=retarget)
    bridge = Bridge(corrector, sink, retarget, physics)

    motion = synthetic_penetrating_motion(retarget, n=n)
    pairs = contract.COLLIDING_PAIRS
    pen_in = physics.get_penetration_depths_from_quats(motion[:, :3], motion[:, 3:], pairs) * 100.0   # [n, 4] cm
    assert float(pen_in.min()) >= min_input_pen_cm, \
        f"합성 입력의 관통이 부족합니다 (최소 {float(pen_in.min()):.2f} cm) — 관통 제거 검사가 공허해집니다"
    hips_abs = torch.tensor([0.1, 0.9, -0.2])
    R_yaw = rt.q_axis_angle(rt._Y, math.radians(yaw_deg))
    hi = 3 + 4 * contract.BONE_INDEX['Hips']

    def _as_tracker(i):
        """합성 프레임 i 를 월드 yaw 로 돌리고 절대 위치로 옮긴 리그 프레임 (트래커가 보낼 것)."""
        f = motion[i].clone()
        f[hi:hi + 4] = rt.qmul(R_yaw, f[hi:hi + 4])
        f[:3] = rt.qrot(R_yaw, f[:3]) + hips_abs
        return f
    got = 0
    corrected_out = []                                        # 수신측에 도착한 보정 프레임 (VRM 규약)
    finger = ("/VMC/Ext/Bone/Pos", ["LeftIndexProximal", 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.995])
    blend = [("/VMC/Ext/Blend/Val", ["Joy", 0.7]), ("/VMC/Ext/Blend/Apply", [])]
    def _wait(rx, timeout=0.5):
        """본 갱신(dirty)이 도착할 때까지 폴링. UDP 는 비동기라 고정 sleep 으로는 못 기다린다."""
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            got = rx.take()
            if got[0] is not None and got[4]:
                return got
            time.sleep(0.001)
        return rx.take()

    time.sleep(0.2)                                           # 서버 스레드 기동
    for i in range(n):
        tracker.send(_as_tracker(i), extra=[finger] + blend)  # 트래커가 리그 값을 VRM 으로 바꿔 보낸다
        frame_vrm, uc, pas, root, dirty, missing = _wait(rx_in)
        assert frame_vrm is not None and dirty, f"입력 수신 실패 (빠진 본 {missing}, 메시지 {rx_in.n_msgs})"
        y, info = bridge.step(frame_vrm, uc, pas, root)
        out, _, pas2, _, dirty2, _ = _wait(rx_out)
        if out is not None and dirty2:
            got += 1
            last_out, last_pass = out, pas2
            if y is not None and info.get("emitted_stage") == "corrected":
                corrected_out.append(out)
    rx_in.stop()
    rx_out.stop()
    assert got >= n - contract.SEQ_LEN, f"출력 수신 {got}/{n}"
    # 0) 보정 예외가 없었는가 — Bridge.step 은 예외를 원본 통과로 흡수하므로 세지 않으면 통과해 버린다
    assert bridge.errors == 0, f"보정 예외 {bridge.errors}건 (원본 통과로 흡수됨)"
    # 1) 출력이 수신측에서 유효한 프레임인가
    probs = contract.validate_frame(retarget.vrm_to_rig(last_out), physics=physics)
    assert not probs, f"출력 프레임 계약 위반: {probs}"
    # 2) Hips 절대 위치가 되돌아왔는가 (정규화가 출력에 새지 않았는가)
    want = _as_tracker(n - 1 - corrector.delay)[:3]           # 출력은 입력보다 delay(=emit_lag) 프레임 늦다
    assert (last_out[:3] - want).abs().max() < 1e-4, \
        f"Hips 위치 불일치 {last_out[:3].tolist()} vs {want.tolist()}"
    # 2b) 정규화가 트래커의 월드 yaw 를 되돌려 모델이 '원래 합성 프레임'을 보았는가
    #     (모델 입력 = 링버퍼의 마지막 프레임. Hips 는 첫 프레임 기준 상대값이어야 한다)
    seen = corrector._buf[-1]
    ref = motion[n - 1].clone()
    ref[:3] = ref[:3] - motion[0][:3]
    q_err = float((1 - (seen[3:].reshape(21, 4) * ref[3:].reshape(21, 4)).sum(-1).abs()).max())
    p_err_cm = float((seen[:3] - ref[:3]).norm() * 100)
    assert q_err < 1e-4 and p_err_cm < 0.3, \
        (f"정규화가 yaw {yaw_deg}° 를 되돌리지 못했습니다: 쿼터니언 오차 {q_err:.2e}, "
         f"Hips {p_err_cm:.3f} cm (세션 yaw {bridge.norm.yaw_deg:.2f}°)")
    # 2c) 관통 제거: 수신측에 도착한 보정 프레임의 4쌍 관통 (월드 yaw·Hips 이동은 깊이와 무관)
    n_corr_min = n - contract.SEQ_LEN - corrector.delay
    assert len(corrected_out) >= n_corr_min, f"보정 프레임 수신 {len(corrected_out)} (기대 ≥ {n_corr_min})"
    co = torch.stack([retarget.vrm_to_rig(f) for f in corrected_out])
    pen_out = physics.get_penetration_depths_from_quats(co[:, :3], co[:, 3:], pairs) * 100.0
    residual = float(pen_out.max())
    assert residual <= max_residual_cm, \
        (f"보정 프레임에 관통이 남았습니다: 쌍별 max {[round(float(x), 3) for x in pen_out.amax(0)]} cm "
         f"(입력 {float(pen_in.min()):.1f}~{float(pen_in.max()):.1f} cm, K 깎임 {corrector.summary()['k_capped']}회)")
    # 3) pass-through (손가락·표정) 가 그대로 도착했는가
    names = [(a, v[0] if v else None) for a, v in last_pass]
    assert ("/VMC/Ext/Bone/Pos", "LeftIndexProximal") in names, f"손가락 pass-through 누락: {names}"
    assert ("/VMC/Ext/Blend/Val", "Joy") in names, f"표정 pass-through 누락: {names}"
    # 4) 우리 21본은 pass-through 에 섞이지 않았는가 (이중 전송 방지)
    assert not any(a == "/VMC/Ext/Bone/Pos" and v in BODY_BONES for a, v in names), "21본이 이중 전송됨"
    # 5) 아무 보정도 없는 입력이면 (워밍업 구간) 출력 회전이 입력과 같아야 한다 — 변환 왕복 검증
    print(f"[loopback] 관통 제거: 입력 4쌍 관통 {float(pen_in.min()):.1f}~{float(pen_in.max()):.1f} cm (전 프레임) → "
          f"수신된 보정 프레임 {len(corrected_out)}개의 4쌍 관통 max {residual:.3f} cm")
    print(f"[loopback] OK: {got} 프레임 왕복, 오류 {bridge.errors}건, 트래커 yaw {yaw_deg:.1f}° → "
          f"세션 정규화 yaw {bridge.norm.yaw_deg:.2f}° (모델 입력 오차: 쿼터니언 {q_err:.1e}, Hips {p_err_cm:.3f} cm)")
    print("  " + sink.report().replace("\n", "\n  "))
    return True


# =====================================================================
def _kbhit_key():
    try:
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getwch().lower()
    except ImportError:
        pass
    return None


def main(argv=None):
    if argv is None and len(sys.argv) == 1:          # Run Python File (인자 없음) → [실행 설정] 블록
        argv = _argv_from_settings()
        print("[실행 설정] python AI_Streaming/vmc_bridge.py " + " ".join(argv))
    ap = argparse.ArgumentParser(description="VMC 브리지: 트래커 → 보정 → Warudo")
    ap.add_argument("--listen-port", type=int, default=39540, help="트래커가 보내는 포트")
    ap.add_argument("--send-host", default="127.0.0.1")
    ap.add_argument("--send-port", type=int, default=39539, help="Warudo VMC 수신 포트 (기본 39539)")
    ap.add_argument("--fps", type=float, default=contract.FPS)
    ap.add_argument("--dry-run", action="store_true", help="전송하지 않는다 (수신·보정만)")
    ap.add_argument("--replay", default=None, choices=["clean", "transient", "persistent", "legacy80"],
                    help="트래커 대신 held-out 데이터셋을 재생해 보낸다 (Warudo 에서 변환을 눈으로 확인)")
    ap.add_argument("--replay-hips-height", type=float, default=0.9,
                    help="--replay 시 Hips 절대 높이(m). 아바타가 뜨거나 잠기면 조정")
    ap.add_argument("--replay-no-level", action="store_true",
                    help="--replay 시 경사 바닥을 펴지 않는다 (원본 그대로 — 아바타가 비스듬히 올라간다)")
    ap.add_argument("--no-fold-upperchest", action="store_true")
    ap.add_argument("--no-projection", action="store_true")
    ap.add_argument("--no-lowpass", action="store_true")
    ap.add_argument("--no-adaptive-k", action="store_true")
    ap.add_argument("--no-retarget", action="store_true",
                    help="[디버그] 리그 값을 변환 없이 보낸다 — 아바타가 뒤틀리는 것이 정상")
    ap.add_argument("--loopback-test", action="store_true", help="Warudo·데이터셋 없이 소켓 왕복 + 관통 제거 자체 검사 (합성 관통 모션)")
    ap.add_argument("--stats-every", type=float, default=5.0)
    ap.add_argument("--idle-after", type=float, default=0.5,
                    help="트래커 수신이 이 시간(초) 이상 없으면 송신을 멈춘다 (held 프레임을 계속 보내지 않는다)")
    ap.add_argument("--reset-after", type=float, default=2.0,
                    help="트래커가 이 시간(초) 이상 끊겼다 돌아오면 세션(정규화 기준·링버퍼)을 리셋한다")
    ap.add_argument("--log-csv", default=None,
                    help="프레임별 실측 기록 CSV (t, stage, ms, k_used, k_capped, proj_residual_cm, hold, error)")
    ap.add_argument("--status-port", type=int, default=0,
                    help="상태·제어 HTTP 포트 (127.0.0.1 전용, 0 = 끔). server.py 가 이 포트로 감시한다")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU 스레드 수 (0 = torch 기본값). 상시 실행에서는 Warudo 와 CPU 를 나눠 쓴다")
    ap.add_argument("--no-keys", action="store_true", help="키보드 입력(b/q)을 읽지 않는다 (서비스 실행)")
    ap.add_argument("--emit-lag", type=int, default=2,
                    help="윈도우 끝에서 몇 프레임 앞을 내보낼지 (기본 2 = +67ms, 0 = 끝자리 — 튐이 크다)")
    ap.add_argument("--hold-repush", action="store_true",
                    help="새 트래커 프레임이 없는 틱에도 같은 프레임을 모델에 다시 넣는다 (2026-10-07 이전 동작)")
    args = ap.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    physics = contract.make_physics("cpu")
    retarget = rt.Retarget()
    ckpt = paths.checkpoint_path()
    corrector = StreamingCorrector(ckpt, physics=physics, validate_every=90,
                                   lp_enabled=not args.no_lowpass, proj_enabled=not args.no_projection,
                                   adaptive_k=not args.no_adaptive_k, emit_lag=args.emit_lag)
    print(f"체크포인트: {os.path.relpath(ckpt, paths.ROOT)}")
    print(f"출력 지연: {corrector.delay}프레임 (emit_lag {corrector.emit_lag}, +{corrector.delay * 1000.0 / args.fps:.0f}ms)  "
          f"| hold 틱: {'같은 프레임 재주입(옛 동작)' if args.hold_repush else '모델에 넣지 않음'}")

    if args.loopback_test:
        return 0 if loopback_test(corrector, retarget, physics) else 1

    client = None
    if not args.dry_run:
        from pythonosc.udp_client import SimpleUDPClient
        client = SimpleUDPClient(args.send_host, args.send_port)
    sink = sinks.VmcOscSink(client=client, retarget=(False if args.no_retarget else retarget))
    bridge = Bridge(corrector, sink, retarget, physics, fold_uc=not args.no_fold_upperchest,
                    gap_reset_s=float("inf"))          # 리셋은 아래 루프가 수신 시각으로 판단한다
    if args.no_retarget:
        bridge.norm.retarget = _IdentityRetarget()

    period = 1.0 / args.fps
    print(f"전송: {'DRY-RUN' if client is None else f'{args.send_host}:{args.send_port}'}  "
          f"{args.fps:.0f}fps  파이프라인: 모델{'' if args.no_lowpass else '→저역통과'}"
          f"{'' if args.no_projection else '→사영'}   [b]=bypass 토글  [q]=종료")

    if args.replay:
        src = iter(ReplayAsTracker(args.replay, retarget, physics, hips_height=args.replay_hips_height,
                                   level=not args.replay_no_level))
        rx = None
        print(f"입력: 데이터셋 재생 ({args.replay}), Hips 높이 {args.replay_hips_height} m")
    else:
        rx = VmcReceiver(args.listen_port).start()
        print(f"입력: VMC 수신 포트 {args.listen_port} — 트래커의 송신 대상을 이 포트로 맞추세요")

    log = None
    if args.log_csv:
        import csv
        log_f = open(args.log_csv, "w", newline="", encoding="utf-8")
        log = csv.writer(log_f)
        log.writerow(["t_s", "frame_in", "emitted_stage", "ms", "k_used", "k_capped",
                      "proj_residual_cm", "proj_frames_touched", "hold", "error", "pen_in_cm", "pen_out_cm"])

    ms = deque(maxlen=600)
    pen = PenetrationMeter(retarget, physics)
    pen_last = {}                    # 상태 HTTP 용: 직전 통계 창의 관통 (pop 된 값)
    t_start = time.perf_counter()
    next_t = time.perf_counter()
    t_stats = next_t
    t_pub = 0.0
    n_hold = n_nodata = 0
    idle = False                     # 트래커 무신호 상태
    idle_since = None

    status = None
    if args.status_port:
        import control
        status = control.StatusServer(args.status_port).start()
        print(f"상태·제어: http://127.0.0.1:{args.status_port}/status  (/health, POST /bypass /reset /shutdown)")

    def _status_dict():
        s = sorted(ms)
        summ = corrector.summary()
        return {
            "ok": True, "pid": os.getpid(), "uptime_s": round(time.perf_counter() - t_start, 1),
            "checkpoint": os.path.relpath(ckpt, paths.ROOT),
            "input": f"replay:{args.replay}" if args.replay else f"udp:{args.listen_port}",
            "send": "dry-run" if client is None else f"{args.send_host}:{args.send_port}",
            "frames_in": bridge.frames, "frames_sent": bridge.sent,
            "tracker_frames": rx.n_frames if rx else None,
            "tracker_silent_s": (round(time.perf_counter() - rx.last_rx, 2) if rx and rx.last_rx else None),
            "idle": idle, "bypass": corrector.bypass,
            "step_ms_p95_recent600": round(s[int(0.95 * (len(s) - 1))], 3) if s else 0.0,
            "step_ms_max_recent600": round(max(s), 3) if s else 0.0,
            "over_budget_pct_recent": round(summ["recent_over_budget_pct"], 3),
            "over_budget_pct_total": round(summ["over_budget_pct"], 3),
            "k_capped": summ["k_capped"], "hold": n_hold,
            "emit_lag": corrector.emit_lag, "hold_repush": args.hold_repush,
            "errors": bridge.errors, "send_errors": bridge.send_errors,
            "last_send_error": bridge.last_send_error,
            "torch_threads": torch.get_num_threads(),
            "pen_in_max_cm_last_window": round(pen_last.get("in", 0.0), 3),
            "pen_out_max_cm_last_window": round(pen_last.get("out", 0.0), 3),
            "pen_in_max_cm_session": round(pen.session["in"], 3),
            "pen_out_max_cm_session": round(pen.session["out"], 3),
            "raw_frames_session": pen.session["n_raw"],
        }

    stop = False
    try:
        while not stop:
            if status is not None:
                status.beat()
                for cmd, arg in status.drain():
                    if cmd == "bypass":
                        corrector.bypass = (not corrector.bypass) if arg is None else bool(arg)
                        print(f"  [control] bypass = {corrector.bypass}")
                    elif cmd == "reset":
                        bridge.reset_session()
                        print("  [control] 세션 리셋")
                    elif cmd == "shutdown":
                        print("  [control] 종료 요청")
                        stop = True
                if stop:
                    break
                if time.perf_counter() - t_pub >= 1.0:
                    status.publish(_status_dict())
                    t_pub = time.perf_counter()
            k = None if args.no_keys else _kbhit_key()
            if k == "q":
                break
            if k == "b":
                corrector.bypass = not corrector.bypass
                print(f"  bypass = {corrector.bypass}")

            if rx is None:
                frame, uc, pas, root = next(src), None, [], None
                dirty = True
            else:
                frame, uc, pas, root, dirty, missing = rx.take()
                silent = time.perf_counter() - rx.last_rx if rx.last_rx else float("inf")
                if frame is not None and silent > args.idle_after:
                    # 트래커가 멈췄다 — held 프레임을 Warudo 로 계속 보내지 않는다 (Warudo 는 마지막 포즈 유지)
                    if not idle:
                        # 끊김의 시작 = 마지막 수신 시각. '무신호로 판정한 시각'(= 그보다 idle_after 뒤)으로
                        # 두면 끊김을 idle_after 만큼 짧게 재서 리셋이 2.0s 가 아니라 2.5s 에야 걸렸다.
                        idle, idle_since = True, rx.last_rx
                        print(f"  트래커 무신호 {args.idle_after:.1f}s — 송신 일시정지")
                    next_t += period
                    time.sleep(max(0.0, next_t - time.perf_counter()))
                    continue
                if idle and frame is not None:
                    gap = time.perf_counter() - idle_since
                    idle = False
                    if gap >= args.reset_after:
                        bridge.reset_session()
                        print(f"  트래커 재개 (끊김 {gap:.1f}s) — 세션 리셋, 첫 1초는 원본 통과")
                    else:
                        print(f"  트래커 재개 (끊김 {gap:.1f}s)")
                if frame is None:
                    n_nodata += 1
                    if n_nodata % int(args.fps * 5) == 1:
                        print(f"  수신 대기… (메시지 {rx.n_msgs}개, 빠진 본 {missing[:5]})")
                    next_t += period
                    time.sleep(max(0.0, next_t - time.perf_counter()))
                    continue
                if not dirty:
                    n_hold += 1
                    if not args.hold_repush:
                        # 새 트래커 프레임이 없는 틱 — 같은 프레임을 모델 창에 다시 넣으면 그 '1프레임 정지'가
                        #   창에 머무는 30프레임 동안 출력을 흔든다(실측 p95 2.0, max 8.9cm). 넣지도 보내지도
                        #   않는다(Warudo 는 마지막 포즈 유지 = 트래커 원본을 받는 캐릭터와 같은 동작).
                        #   단, 그 사이 들어온 pass-through(표정·손가락)는 take() 가 이미 꺼냈으므로 버리지 않고
                        #   직전 출력 포즈에 실어 보낸다.
                        if pas and bridge.last_out_vrm is not None:
                            try:
                                sink.send(bridge.last_out_vrm, extra=pas, root=root, already_vrm=True)
                            except OSError as e:
                                bridge.send_errors += 1
                                bridge.last_send_error = f"{type(e).__name__}: {e}"
                        if log:
                            log.writerow([f"{time.perf_counter() - t_start:.4f}", bridge.frames, "hold_skip",
                                          "", "", "", "", "", 1, ""])
                        next_t += period
                        time.sleep(max(0.0, next_t - time.perf_counter()))
                        continue
            t0 = time.perf_counter()
            y, info = bridge.step(frame, uc, pas, root)
            step_ms = (time.perf_counter() - t0) * 1000.0
            ms.append(step_ms)
            # 관통 실측은 step 시간 밖 — 처리 ms·예산초과 숫자는 예전과 같은 것을 잰다
            emitted = info.get("emitted_stage") or info.get("stage")
            p_in, p_out = pen.update(bridge.last_in_fold, bridge.last_out_fold, emitted)
            if log:
                log.writerow([f"{t0 - t_start:.4f}", bridge.frames, emitted,
                              f"{step_ms:.3f}", info.get("k_used", ""), int(bool(info.get("k_capped"))),
                              f"{info.get('proj_residual_max_cm', 0.0):.3f}",
                              info.get("proj_frames_touched", ""), int(not dirty), info.get("error", ""),
                              "" if p_in is None else f"{p_in:.3f}", "" if p_out is None else f"{p_out:.3f}"])
                log_f.flush()                                 # 강제 종료돼도 행이 남도록
            if info.get("stage") == "error" and bridge.consecutive_errors <= 3:
                print(f"  ⚠️ 보정 예외 → 원본 통과: {info['error']}")
            if info.get("send_error") and (bridge.send_errors in (1, 10, 100) or bridge.send_errors % 1000 == 0):
                print(f"  ⚠️ 송신 실패 {bridge.send_errors}회: {info['send_error']}")

            now = time.perf_counter()
            if now - t_stats >= args.stats_every:
                s = sorted(ms)
                p95 = s[int(0.95 * (len(s) - 1))] if s else 0.0
                summ = corrector.summary()
                # 예산초과는 최근 창(STATS_WINDOW) 기준 — 누적 %는 오래 돌수록 최근 스파이크를 묻는다.
                print(f"  in {bridge.frames} out {bridge.sent} | 처리 p95 {p95:5.1f} ms max {max(s) if s else 0:5.1f} "
                      f"| 예산초과 {summ['recent_over_budget_pct']:.1f}% | K깎임 {summ['k_capped']} "
                      f"| hold {n_hold} | 오류 {bridge.errors}"
                      + (f" 송신실패 {bridge.send_errors}" if bridge.send_errors else "")
                      + f" | bypass={corrector.bypass}"
                      + (f" | 트래커 {rx.n_frames}f" if rx else ""))
                # 관통은 이번 통계 창(stats_every)의 max — 누적 max 는 한 번 깊은 입력이 지나가면 고정된다
                pen_last = pen.pop_window()
                print(f"      {PenetrationMeter.format(pen_last)}  [최근 {args.stats_every:.0f}s]")
                t_stats = now
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        if status is not None:
            status.stop()
        if rx is not None:
            rx.stop()
        if log:
            log_f.close()
            print(f"프레임 로그: {args.log_csv}")
        print(f"세션 {PenetrationMeter.format(pen.session)}")
        print(sink.report())
    return 0


class _IdentityRetarget:
    @staticmethod
    def rig_to_vrm(f):
        return torch.as_tensor(f, dtype=torch.float32)

    @staticmethod
    def vrm_to_rig(f):
        return torch.as_tensor(f, dtype=torch.float32)


if __name__ == "__main__":
    sys.exit(main())
