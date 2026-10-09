"""출력부 — 보정된 프레임을 내보내는 곳.

[VmcOscSink 가 Warudo 로 보내는 것 (VMC 프로토콜, OSC/UDP, 기본 포트 39539)]
  한 프레임 = OSC 번들 하나:
    /VMC/Ext/OK        1 3 0 1            (loaded, Calibrated, Normal, tracking OK)  — v2.7 형식
    /VMC/Ext/Root/Pos  "root" 0 0 0 0 0 0 1   (또는 트래커 값 pass-through)
    /VMC/Ext/Bone/Pos  <본> px py pz qx qy qz qw   × 21   — **Hips 만 위치를 싣는다**
    (pass-through: 손가락·눈·UpperChest 등 우리 21본 밖의 본, /VMC/Ext/Blend/Val, Blend/Apply ...)
    /VMC/Ext/T         <초>
  본 이름은 Unity HumanBodyBones 철자 그대로(이 프로젝트의 21개 이름이 이미 그 철자다).

[가장 중요한 변환 — 리그 로컬 -> VRM 로컬]
  Warudo 는 "T-포즈에서 모든 본 회전이 0 인 정규화 모델"을 전제한다. 이 프로젝트의 텐서에 든
  쿼터니언은 FBX 리그 로컬 회전이라 그 규약이 아니다. `retarget.Retarget.rig_to_vrm` 을 거치지
  않은 값을 보내면 아바타가 뒤틀린다. 그래서 이 싱크는 **기본으로 변환을 켠다**
  (retarget=False 는 배선 디버그용).

[Hips 위치]
  데이터셋의 Hips 위치는 '첫 프레임 기준 상대값'(≈0)이다. 그대로 보내면 아바타 골반이 월드
  원점에 놓여 바닥에 파묻힌다. 라이브에서는 브리지가 트래커의 절대 위치로 되돌려 넘기고,
  데이터 재생(run_sim/--replay)에서는 hips_offset 으로 높이를 더한다.

[전송 모드]
  client=SimpleUDPClient  -> 프레임당 번들 1개로 실제 전송 (권장)
  send_fn(addr, args)     -> 메시지 단위로 호출 (테스트/커스텀)
  둘 다 없음              -> DRY-RUN: 만들기만 하고 세지만 보내지 않는다 (기본)
"""

import time

import paths  # noqa: F401

import torch

import contract


class NullSink:
    """아무것도 하지 않는다. 지연만 측정할 때 쓴다."""

    def __init__(self):
        self.n = 0

    def send(self, frame87, meta=None):
        self.n += 1

    def close(self):
        pass


class RecordSink:
    """보낸 프레임을 메모리에 모은다. 시뮬레이션 후 품질 지표를 계산하려고 쓴다."""

    def __init__(self):
        self.frames = []

    def send(self, frame87, meta=None):
        self.frames.append(torch.as_tensor(frame87).detach().clone())

    def tensor(self):
        return torch.stack(self.frames, dim=0) if self.frames else torch.empty(0, contract.FRAME_DIM)

    def close(self):
        pass


class VmcOscSink:
    """VMC(OSC) 프레임을 만들고(선택적으로) 보낸다. 기본은 DRY-RUN.

    예) python-osc 로 Warudo 에 실제 전송:
        from pythonosc.udp_client import SimpleUDPClient
        sink = VmcOscSink(client=SimpleUDPClient("127.0.0.1", 39539))
    """

    BONE_ADDR = "/VMC/Ext/Bone/Pos"
    ROOT_ADDR = "/VMC/Ext/Root/Pos"
    OK_ADDR = "/VMC/Ext/OK"
    T_ADDR = "/VMC/Ext/T"
    OK_ARGS = [1, 3, 0, 1]      # loaded=1, calibration_state=3(Calibrated), mode=0(Normal), tracking=1(OK)

    def __init__(self, send_fn=None, client=None, retarget=None, send_root=True,
                 hips_offset=(0.0, 0.0, 0.0), send_ok=True, send_time=True):
        """
        retarget : retarget.Retarget 인스턴스 / None(=기본 rig_tpose.json 로 생성) / False(변환 끔, 디버그)
        client   : pythonosc.udp_client.SimpleUDPClient — 프레임당 번들 1개 전송
        send_fn  : (addr, args) 콜백 — 메시지 단위 전송
        hips_offset : Hips 위치에 더할 (x,y,z) m
        """
        self.send_fn = send_fn
        self.client = client
        self.send_root = send_root
        self.hips_offset = torch.tensor(hips_offset, dtype=torch.float32)
        self.send_ok = send_ok
        self.send_time = send_time
        if retarget is None:
            import retarget as _rt
            retarget = _rt.Retarget()
        self.retarget = retarget or None
        self.n_frames = 0
        self.n_messages = 0
        self.n_dropped = 0
        self.first_packet = None
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------
    def build(self, frame87, extra=(), root=None, already_vrm=False):
        """[87](리그 규약) -> [(address, args), ...]. 프레임 하나의 메시지 목록.

        extra : 함께 실을 pass-through 메시지 [(addr, args), ...] (손가락·표정 등)
        root  : /VMC/Ext/Root/Pos 인자 목록 (None 이면 단위 root)
        already_vrm : True 면 frame87 이 이미 VRM 규약이라 리그->VRM 변환을 건너뛴다
                      (vmc_bridge 처럼 호출자가 변환을 직접 한 경우).
        """
        f = torch.as_tensor(frame87, dtype=torch.float32).reshape(contract.FRAME_DIM)
        if not torch.isfinite(f).all():
            raise ValueError("NaN/Inf 프레임은 보낼 수 없습니다")
        if self.retarget is not None and not already_vrm:
            f = self.retarget.rig_to_vrm(f)
        f = contract.normalize_quats(f)             # 수신측이 단위 쿼터니언을 가정한다
        hips, quats = contract.unpack_frame(f)
        hips = hips + self.hips_offset
        msgs = []
        if self.send_ok:
            msgs.append((self.OK_ADDR, list(self.OK_ARGS)))
        if self.send_root:
            msgs.append((self.ROOT_ADDR, list(root) if root is not None
                         else ["root", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        for name in contract.BONE_ORDER:
            q = [float(x) for x in quats[name]]
            p = [float(x) for x in hips] if name == "Hips" else [0.0, 0.0, 0.0]
            msgs.append((self.BONE_ADDR, [name, p[0], p[1], p[2], q[0], q[1], q[2], q[3]]))
        msgs.extend((a, list(v)) for a, v in extra)
        if self.send_time:
            msgs.append((self.T_ADDR, [float(time.perf_counter() - self._t0)]))
        return msgs

    def send(self, frame87, meta=None, extra=(), root=None, already_vrm=False):
        try:
            msgs = self.build(frame87, extra=extra, root=root, already_vrm=already_vrm)
        except ValueError:
            self.n_dropped += 1
            return
        if self.first_packet is None:
            self.first_packet = msgs[:4]
        self.n_frames += 1
        self.n_messages += len(msgs)
        if self.client is not None:
            self.client.send(self.bundle(msgs))
        elif self.send_fn is not None:
            for addr, args in msgs:
                self.send_fn(addr, args)

    @staticmethod
    def bundle(msgs):
        """[(addr, args)] -> OSC 번들 (python-osc)."""
        from pythonosc import osc_bundle_builder, osc_message_builder
        bb = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
        for addr, args in msgs:
            mb = osc_message_builder.OscMessageBuilder(address=addr)
            for a in args:
                mb.add_arg(a)
            bb.add_content(mb.build())
        return bb.build()

    def close(self):
        pass

    def report(self):
        if self.client is not None:
            mode = f"실제 전송 -> {self.client._address}:{self.client._port}"
        elif self.send_fn is not None:
            mode = "send_fn 전송"
        else:
            mode = "DRY-RUN (전송 안 함)"
        rt = "리그->VRM 변환 ON" if self.retarget is not None else "변환 OFF(리그 값 그대로 — 디버그 전용)"
        return (f"VmcOscSink[{mode}] {rt} frames={self.n_frames} messages={self.n_messages} "
                f"dropped={self.n_dropped}\n  첫 패킷 예시: {self.first_packet}")
