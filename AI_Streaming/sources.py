"""입력원 — 라이브 스트림을 '재현'하는 시뮬레이터.

[왜 시뮬레이터부터인가]
  실시간 스트림은 데이터가 실시간으로 들고 난다. 실제 송신을 붙이기 전에
  **같은 형태(한 프레임씩, 시계에 맞춰)** 로 흘려보내면서 배선·지연·품질을 먼저 확인해야 한다.
  여기서 쓰는 모션은 **held-out(test) 스플릿**이다 — 학습에 쓴 파일로 시뮬레이션하면
  결과가 낙관적으로 오염된다.

[시나리오]
  clean      : 원본 그대로. do-no-harm 검사용 (모델이 멀쩡한 입력을 건드리는지).
  transient  : 30프레임 창 안에서 들어왔다 나가는 손상 (corruption.inject_transient)
  persistent : 창 전체에 지속되는 손상 (corruption.inject_persistent)
  legacy80   : 고정 80° 팔 주입 = **학습 분포 밖(OOD) 프로브** (evaluate.inject_arm_collision)

  ⚠️ 주입기는 30프레임 창 단위로 동작한다. 그래서 이 소스도 파일을 30프레임 청크로 잘라
  청크마다 주입한다. 즉 라이브에서 "손상이 1초 단위로 들어왔다 나간다"에 해당한다.

[실제 트래커를 붙일 때]
  이 파일의 `__iter__` 가 내놓는 것과 같은 (입력 프레임 [87], 참조 프레임 or None) 을
  내놓는 클래스를 하나 더 만들면 된다. 수신·이름 매핑·쿼터니언 규약 변환은
  contract.pack_frame() 하나로 끝난다.
"""

import time

import paths  # noqa: F401

import torch

import contract

_FEET = ('LeftFoot', 'RightFoot', 'LeftToes', 'RightToes')


def level_ground(motion, physics, iters=3, ridge=1e-3):
    """경사진 바닥 위의 이동을 평지로 편다. Hips 높이(y)만 바꾼 사본을 돌려준다.

    데이터셋(원본 CSV 부터) 걷기·달리기 파일 다수가 기울어진 바닥 위에 있다: 몸(Hips 회전)은
    똑바로 선 채 Hips 위치만 비탈을 따라 오른다. 2026-10-06 held-out 실측(이동 ≥1m 101파일):
    기울기 중앙값 12.7°, 최대 ~48°, 접지 높이 범위 ~1m. Warudo 에서는 아바타가 허공으로
    비스듬히 걸어 올라가는 것처럼 보인다.

    방법: 프레임별 최저 발 높이 h 를 Hips 수평 위치 (x, z) 의 평면 a·x + b·z + c 로 적합한다.
    접지 쪽(잔차 아래 절반)만 다시 적합하기를 반복해 점프·발차기를 배제하고, 그 평면 기울기만큼
    Hips y 를 내린다(첫 프레임 높이는 불변 → --hips-height 의 의미 유지).
    평면 적합 후 접지 높이 범위는 4~8cm(같은 실측). 제자리 동작은 보정량이 cm 이하라 사실상 무변화.
    """
    m = motion.clone()
    gp = physics.forward_kinematics(m[:, :3], m[:, 3:])
    low = torch.stack([gp[b][..., 1].reshape(-1) for b in _FEET if b in gp], -1).min(-1).values.double()
    x = (m[:, 0] - m[0, 0]).double()
    z = (m[:, 2] - m[0, 2]).double()
    A = torch.stack([x, z, torch.ones_like(x)], -1)
    reg = torch.diag(torch.tensor([ridge, ridge, 0.0], dtype=torch.float64))   # 직선 이동이면 옆 기울기가 부정 → 0 쪽으로
    keep = torch.ones_like(x, dtype=torch.bool)
    for _ in range(iters):
        Ak, hk = A[keep], low[keep]
        sol = torch.linalg.solve(Ak.T @ Ak + reg * len(hk), Ak.T @ hk)
        r = low - A @ sol
        keep = r <= r.quantile(0.5)
    m[:, 1] = m[:, 1] - (A[:, :2] @ sol[:2]).to(m.dtype)
    return m


class ReplaySource:
    """held-out .pt 파일들을 실시간처럼 한 프레임씩 흘려보낸다."""

    def __init__(self, files, scenario="clean", fps=contract.FPS, pace=False,
                 seed=20260707, physics=None, max_frames=0, loop=False, level=False):
        # level: 경사 바닥을 평지로 편다 (level_ground). Warudo 로 '보여주는' 경로(vmc_bridge --replay,
        #   tracker_sim)만 켠다 — 기본 False 라 run_sim·selftest 의 실측 입력은 예전과 비트 동일.
        self.level = level
        self.files = list(files)
        if not self.files:
            raise ValueError("재생할 .pt 파일이 없습니다.")
        self.scenario = scenario
        self.fps = fps
        self.pace = pace
        self.seed = seed
        self.max_frames = max_frames
        self.loop = loop
        self.physics = physics if physics is not None else contract.make_physics("cpu")

        self._cfg = None
        self._rng = None
        if scenario in ("transient", "persistent"):
            import corruption
            import random
            self._corruption = corruption
            self._cfg = corruption.make_cfg()
            self._rng = random.Random(seed)
        elif scenario == "legacy80":
            import importlib.util
            import os
            spec = importlib.util.spec_from_file_location(
                "_ai_model_evaluate_for_inject", os.path.join(paths.AI_MODEL_DIR, "evaluate.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._inject_arm = mod.inject_arm_collision
        elif scenario != "clean":
            raise ValueError(f"알 수 없는 시나리오: {scenario} "
                             f"(clean/transient/persistent/legacy80)")

    # ------------------------------------------------------------------
    def _corrupt_chunk(self, chunk):
        """[S, 87] 청크에 시나리오별 손상을 주입한 사본을 돌려준다."""
        if self.scenario == "clean":
            return chunk
        if self.scenario == "transient":
            return self._corruption.inject_transient(
                chunk, self.physics, self._cfg, self._rng, contract.COLLIDING_PAIRS)[0]
        if self.scenario == "persistent":
            return self._corruption.inject_persistent(
                chunk, self.physics, self._cfg, self._rng, contract.COLLIDING_PAIRS)[0]
        if self.scenario == "legacy80":
            return self._inject_arm(chunk, angle_deg=80.0)
        raise AssertionError(self.scenario)

    # ------------------------------------------------------------------
    def __iter__(self):
        """(입력 프레임 [87], 클린 참조 프레임 [87]) 를 순서대로 내놓는다."""
        n = 0
        period = 1.0 / self.fps
        next_t = time.perf_counter()
        while True:
            for path in self.files:
                motion = torch.load(path)                    # [Frames, 87]
                if self.level:
                    motion = level_ground(motion, self.physics)
                S = contract.SEQ_LEN
                for s in range(0, motion.shape[0] - S + 1, S):
                    clean_chunk = motion[s:s + S]
                    inp_chunk = self._corrupt_chunk(clean_chunk)
                    for i in range(S):
                        if self.pace:
                            next_t += period
                            delay = next_t - time.perf_counter()
                            if delay > 0:
                                time.sleep(delay)
                            else:
                                next_t = time.perf_counter()   # 밀렸으면 따라잡지 않는다
                        yield inp_chunk[i], clean_chunk[i]
                        n += 1
                        if self.max_frames and n >= self.max_frames:
                            return
            if not self.loop:
                return


def describe_scenarios():
    return {
        "clean": "원본 그대로 — do-no-harm 검사",
        "transient": "창 안에서 들어왔다 나가는 손상",
        "persistent": "창 전체에 지속되는 손상",
        "legacy80": "고정 80° 팔 주입 (학습 분포 밖 OOD 프로브)",
    }
