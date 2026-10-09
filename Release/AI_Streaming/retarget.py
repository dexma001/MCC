"""리타깃 — 이 프로젝트의 리그 로컬 회전 ↔ VRM 정규화 로컬 회전 (Warudo 가 받는 형식).

[왜 이 파일이 필요한가 — Warudo 호환의 핵심]
  Warudo 의 VMC 수신은 "T-포즈에서 모든 본의 로컬 회전이 (0,0,0)" 인 정규화 모델(VRM)을
  전제한다. 그런데 이 프로젝트의 87차원 텐서에 들어 있는 21개 로컬 쿼터니언은 그 규약이
  **아니다**: 원본 CSV 는 FBX 리그의 로컬 회전이라 관절 방향(joint orient)이 통째로 들어 있다.
  (예: 정지 자세에서도 Spine ≈ Y축 90°, LeftUpperLeg ≈ Z축 180°, Shoulder ≈ 120°.)
  이 값을 그대로 `/VMC/Ext/Bone/Pos` 로 보내면 아바타가 완전히 뒤틀린다. 반대로 트래커가
  보내는 VRM 규약 회전을 그대로 모델에 넣으면 학습 분포와 전혀 다른 입력이 된다.
  ⇒ 양방향 변환이 반드시 있어야 하고, 이 파일이 그 유일한 지점이다.

[수학]
  리그의 각 본 b 에 대해 T-포즈에서의 전역 회전 G0_b 를 알면(rig_tpose.json),
    리그 로컬 L_b  = inv(G0_parent) · V_b · G0_b
    VRM 로컬 V_b   = G0_parent · L_b · inv(G0_b)
  (Hips 의 parent 는 단위 회전.) 이 변환은 **모든 관절의 월드 위치를 정확히 보존**한다 —
  VRM 쪽 뼈 오프셋을 G0_parent·offset_b 로 두면 두 리그의 FK 가 일치한다(selftest 가 검사).
  각 본의 월드 회전은 Δ_b = G_b·inv(G0_b) 로 서로 독립이라 오차가 사슬을 따라 누적되지 않는다.

  G0 의 '비틀림(twist)' 성분은 기하로 정해지지 않는 자유도다 — Hips/Chest 는 좌우 자식
  오프셋으로 고정되고, 나머지는 데이터셋의 평균 자세에서 최소 회전(swing)으로 옮겨 정한다.
  도출 과정과 검증은 derive_rest_pose.py 에 있다. **G0 가 틀려도 관절 위치는 맞는다**.
  틀리면 본의 '롤'(팔 비틀림·손바닥 방향·머리 yaw)과 모델 입력 분포가 어긋난다.

[좌표계] Unity Y-Up 왼손좌표계 그대로. 쿼터니언 (x,y,z,w). Unity 의 q*v 규약과 같은 회전식.
  FK 로 실측한 리그의 축: 위 = +Y, 왼쪽 = -X (LeftUpperLeg 가 -X), 앞 = +Z (발끝이 +Z).
"""

import json
import os

import paths  # noqa: F401

import torch

import contract
from dataset_pipeline import PARENTS

TPOSE_JSON = os.path.join(paths.HERE, "rig_tpose.json")

_X = torch.tensor([1.0, 0.0, 0.0])
_Y = torch.tensor([0.0, 1.0, 0.0])
_Z = torch.tensor([0.0, 0.0, 1.0])
QID = torch.tensor([0.0, 0.0, 0.0, 1.0])


# =====================================================================
# 쿼터니언 유틸 (physics_module 과 같은 규약; 배치 브로드캐스트 지원)
# =====================================================================
def qmul(a, b):
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], dim=-1)


def qconj(q):
    """단위 쿼터니언의 역."""
    return q * torch.tensor([-1.0, -1.0, -1.0, 1.0], dtype=q.dtype, device=q.device)


def qrot(q, v):
    """v 를 q 로 회전 (Unity q*v)."""
    t = 2.0 * torch.cross(q[..., :3], v, dim=-1)
    return v + q[..., 3:4] * t + torch.cross(q[..., :3], t, dim=-1)


def qnorm(q, eps=1e-8):
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def qangle_deg(q):
    """회전 각도 (deg, 0~360)."""
    w = q[..., 3].abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(w))


def q_axis_angle(axis, angle_rad):
    """축·각 -> 쿼터니언 (배치 없음)."""
    axis = axis / (axis.norm() + 1e-12)
    h = float(angle_rad) * 0.5
    import math
    return torch.cat([axis * math.sin(h), torch.tensor([math.cos(h)], dtype=axis.dtype)])


def q_swing(a, b, eps=1e-9):
    """단위벡터 a 를 b 로 보내는 최소 회전(swing). 배치 지원. a≈-b 이면 임의의 수직축 180°."""
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    c = torch.cross(a, b, dim=-1)
    d = (a * b).sum(-1, keepdim=True)
    q = torch.cat([c, 1.0 + d], dim=-1)
    anti = (1.0 + d).squeeze(-1) < 1e-6
    if anti.any():
        alt = torch.cross(a, _Y.to(a).expand_as(a), dim=-1)
        bad = alt.norm(dim=-1) < 1e-6
        alt = torch.where(bad.unsqueeze(-1), torch.cross(a, _X.to(a).expand_as(a), dim=-1), alt)
        alt = torch.cat([alt / (alt.norm(dim=-1, keepdim=True) + eps), torch.zeros_like(d)], -1)
        q = torch.where(anti.unsqueeze(-1), alt, q)
    return qnorm(q)


def q_mirror_x(q):
    """x=0 평면(좌우) 대칭 회전. (x,y,z,w) -> (x,-y,-z,w)."""
    return q * torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=q.dtype, device=q.device)


# =====================================================================
# 리그 FK (회전 포함). physics_module 은 위치만 돌려주므로 전역 회전은 여기서 계산한다.
# =====================================================================
_ORDER = [b for b in PARENTS]                 # 부모가 먼저 오는 순서 (PARENTS 삽입 순)


def global_rotations(quats_21x4):
    """[..., 21, 4] 로컬 -> {본: [..., 4]} 전역 회전 (physics.forward_kinematics 와 같은 합성)."""
    bi = contract.BONE_INDEX
    G = {'Hips': quats_21x4[..., bi['Hips'], :]}
    for bone in _ORDER:
        p = PARENTS[bone]
        if p is None:
            continue
        G[bone] = qmul(G[p], quats_21x4[..., bi[bone], :])
    return G


def fk_with_offsets(hips_pos, quats_21x4, offsets):
    """offsets = {본: [3]} 를 쓰는 FK. VRM 쪽 리그 검증용 (physics 와 같은 식)."""
    G = global_rotations(quats_21x4)
    P = {'Hips': hips_pos}
    for bone in _ORDER:
        p = PARENTS[bone]
        if p is None:
            continue
        o = offsets[bone].to(hips_pos).expand(*hips_pos.shape[:-1], 3)
        P[bone] = P[p] + qrot(G[p], o)
    return P, G


# =====================================================================
# 리타깃 본체
# =====================================================================
class Retarget:
    """rig_tpose.json 의 G0 로 리그 로컬 ↔ VRM 로컬을 바꾼다. 상태 없음, 결정론."""

    def __init__(self, tpose_path=TPOSE_JSON):
        if not os.path.exists(tpose_path):
            raise FileNotFoundError(
                f"리그 T-포즈 파일이 없습니다: {tpose_path}\n"
                f"  python AI_Streaming/derive_rest_pose.py 로 먼저 만드세요.")
        with open(tpose_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        g0 = self.meta["G0"]
        missing = [b for b in contract.BONE_ORDER if b not in g0]
        if missing:
            raise ValueError(f"rig_tpose.json 에 없는 본: {missing}")
        self.G0 = qnorm(torch.tensor([g0[b] for b in contract.BONE_ORDER], dtype=torch.float32))
        self.G0p = torch.stack([
            self.G0[contract.BONE_INDEX[PARENTS[b]]] if PARENTS[b] is not None else QID
            for b in contract.BONE_ORDER])
        self.G0_inv = qconj(self.G0)
        self.G0p_inv = qconj(self.G0p)
        self.L0 = qmul(self.G0p_inv, self.G0)          # 리그의 T-포즈 로컬 (VRM 단위 회전에 대응)

    # --- 텐서 모양 처리 ---------------------------------------------
    @staticmethod
    def _split(frame):
        f = torch.as_tensor(frame, dtype=torch.float32)
        lead = f.shape[:-1]
        return f[..., :3], f[..., 3:].reshape(*lead, contract.NUM_JOINTS, 4)

    @staticmethod
    def _join(hips, q):
        return torch.cat([hips, q.reshape(*q.shape[:-2], contract.NUM_JOINTS * 4)], dim=-1)

    # --- 변환 --------------------------------------------------------
    def rig_to_vrm(self, frame87):
        """리그 로컬 [.., 87] -> VRM 로컬 [.., 87]. Hips 위치는 그대로."""
        hips, L = self._split(frame87)
        V = qmul(qmul(self.G0p.to(L), L), self.G0_inv.to(L))
        return self._join(hips, qnorm(V))

    def vrm_to_rig(self, frame87):
        """VRM 로컬 [.., 87] -> 리그 로컬 [.., 87]. Hips 위치는 그대로."""
        hips, V = self._split(frame87)
        L = qmul(qmul(self.G0p_inv.to(V), V), self.G0.to(V))
        return self._join(hips, qnorm(L))

    def vrm_offsets(self, physics):
        """VRM 쪽(T-포즈=단위 회전) 리그의 뼈 오프셋 = G0_parent · offset_b. 검증용."""
        out = {}
        for b in contract.BONE_ORDER:
            p = PARENTS[b]
            o = physics.bone_offsets[b]
            out[b] = qrot(self.G0[contract.BONE_INDEX[p]], o) if p is not None else o.clone()
        return out

    def rig_tpose_frame(self, hips=(0.0, 0.0, 0.0)):
        """리그 규약으로 표현한 T-포즈 [87] (VRM 로컬이 전부 단위일 때의 리그 로컬)."""
        return self._join(torch.tensor(hips, dtype=torch.float32), self.L0)
