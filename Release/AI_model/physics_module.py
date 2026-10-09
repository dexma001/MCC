import os
import torch
import torch.nn as nn
import pandas as pd


class DifferentiablePhysics(nn.Module):
    """
    새 데이터 양식(Hips Position 3 + 21개 관절 Local Quaternion 84 = 87)에 맞춘
    미분 가능한 물리 엔진.

    관절별 위치를 데이터에서 직접 받지 않고, 고정 뼈대 오프셋(Standard_BoneOffsets.csv)과
    Local Quaternion으로 순방향 운동학(FK)을 수행하여 관절의 월드 좌표를 복원한 뒤,
    캡슐(선분) 간 최단 거리 기반으로 충돌 Loss를 계산한다.

    좌표계 참고: FK는 데이터 원본 좌표계(Unity Y-Up) 그대로 수행한다. 충돌 '거리'는
    전역 회전/평행이동에 불변이므로 파이썬 시각화용 좌표 변환 없이도 물리적으로 정확하다.
    """

    def __init__(self, parents, bone_radii, offset_csv_path=None):
        super().__init__()
        self.parents = parents
        self.bone_names = sorted(list(parents.keys()))
        self.bone_map = {name: i for i, name in enumerate(self.bone_names)}

        # 뼈 두께(반지름)는 스칼라 상수로 보관 → 디바이스 이동 이슈 없이 threshold 계산에 사용
        self.bone_radii = {k: float(v) for k, v in bone_radii.items()}

        # 고정 뼈대 오프셋(뼈 길이/방향) 로드. 학습 중 변하지 않는 상수.
        self.bone_offsets = self._load_bone_offsets(offset_csv_path)

    # ------------------------------------------------------------------
    # 오프셋 로드
    # ------------------------------------------------------------------
    def _resolve_offset_path(self, offset_csv_path):
        """실행 위치(프로젝트 루트 / AI_model 등)와 무관하게 오프셋 CSV를 찾는다."""
        candidates = []
        if offset_csv_path:
            candidates.append(offset_csv_path)
        here = os.path.dirname(os.path.abspath(__file__))
        candidates += [
            "Sample_Data/Standard_BoneOffsets.csv",
            "../Sample_Data/Standard_BoneOffsets.csv",
            os.path.join(here, "..", "Sample_Data", "Standard_BoneOffsets.csv"),
            os.path.join(here, "Sample_Data", "Standard_BoneOffsets.csv"),
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c
        raise FileNotFoundError(
            "Standard_BoneOffsets.csv를 찾을 수 없습니다. offset_csv_path 인자로 경로를 지정하세요."
        )

    def _load_bone_offsets(self, offset_csv_path):
        path = self._resolve_offset_path(offset_csv_path)
        df = pd.read_csv(path).set_index('BoneName')
        offsets = {}
        for bone in self.bone_names:
            if bone in df.index:
                row = df.loc[bone]
                vec = [float(row['OffsetX']), float(row['OffsetY']), float(row['OffsetZ'])]
            else:
                vec = [0.0, 0.0, 0.0]
            offsets[bone] = torch.tensor(vec, dtype=torch.float32)
        return offsets

    # ------------------------------------------------------------------
    # 쿼터니언 연산 (Unity 좌표계 x, y, z, w 규약)
    # ------------------------------------------------------------------
    def quat_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([x, y, z, w], dim=-1)

    def quat_rotate_vector(self, q, v):
        """쿼터니언 q로 3D 벡터 v를 회전 (Unity의 q * v 규약)."""
        q_xyz = q[..., :3]
        q_w = q[..., 3:4]
        t = 2.0 * torch.cross(q_xyz, v, dim=-1)
        return v + q_w * t + torch.cross(q_xyz, t, dim=-1)

    # ------------------------------------------------------------------
    # 순방향 운동학 (Forward Kinematics)
    # ------------------------------------------------------------------
    def forward_kinematics(self, hips_pos, quats_84):
        """
        hips_pos : [..., 3]  루트(Hips)의 월드 좌표
        quats_84 : [..., 84] 21개 관절의 Local Quaternion (BONE_NAMES 정렬 순서)

        반환: {뼈이름: [..., 3]} 관절별 월드 좌표 딕셔너리
        """
        lead = hips_pos.shape[:-1]
        quats = quats_84.reshape(*lead, len(self.bone_names), 4)
        local_quats = {name: quats[..., self.bone_map[name], :] for name in self.bone_names}

        global_pos = {'Hips': hips_pos}
        global_rot = {'Hips': local_quats['Hips']}

        # PARENTS는 부모가 항상 자식보다 먼저 등장하는 위상 정렬 순서
        for bone, parent in self.parents.items():
            if parent is None:
                continue
            offset = self.bone_offsets[bone].to(hips_pos.device)
            offset = offset.view(*([1] * len(lead)), 3).expand(*lead, 3)
            global_rot[bone] = self.quat_multiply(global_rot[parent], local_quats[bone])
            global_pos[bone] = global_pos[parent] + self.quat_rotate_vector(global_rot[parent], offset)

        return global_pos

    def compute_global_pos_tensor(self, hips_pos, quats_84):
        """FK 결과를 [..., 21, 3] 텐서로 반환 (jitter/MPJPE 계산 편의용)."""
        gp = self.forward_kinematics(hips_pos, quats_84)
        return torch.stack([gp[name] for name in self.bone_names], dim=-2)

    # ------------------------------------------------------------------
    # 미분 가능한 캡슐(선분) 최단 거리 — Lumelsky 알고리즘 텐서화
    # ------------------------------------------------------------------
    def capsule_distance(self, p1, q1, p2, q2):
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

        sD = D
        tD = D

        sN = torch.where(D < SMALL_NUM, torch.zeros_like(D), b * e - c * d)
        tN = torch.where(D < SMALL_NUM, e, a * e - b * d)
        tD = torch.where(D < SMALL_NUM, c, tD)

        s_less_0 = sN < 0.0
        sN = torch.where(s_less_0, torch.zeros_like(sN), sN)
        tN = torch.where(s_less_0, e, tN)
        tD = torch.where(s_less_0, c, tD)

        s_greater_d = sN > sD
        sN = torch.where(s_greater_d, sD, sN)
        tN = torch.where(s_greater_d, e + b, tN)
        tD = torch.where(s_greater_d, c, tD)

        zeros = torch.zeros_like(a)

        t_less_0 = tN < 0.0
        tN = torch.where(t_less_0, torch.zeros_like(tN), tN)
        sN_new_t0 = torch.clamp(-d, min=zeros, max=a)
        sN = torch.where(t_less_0, sN_new_t0, sN)
        sD = torch.where(t_less_0, a, sD)

        t_greater_d = tN > tD
        tN = torch.where(t_greater_d, tD, tN)
        sN_new_t1 = torch.clamp(-d + b, min=zeros, max=a)
        sN = torch.where(t_greater_d, sN_new_t1, sN)
        sD = torch.where(t_greater_d, a, sD)

        safe_sD = torch.clamp(sD, min=SMALL_NUM)
        safe_tD = torch.clamp(tD, min=SMALL_NUM)

        sc = torch.where(torch.abs(sN) < SMALL_NUM, torch.zeros_like(sN), sN / safe_sD)
        tc = torch.where(torch.abs(tN) < SMALL_NUM, torch.zeros_like(tN), tN / safe_tD)

        sc = sc.unsqueeze(-1)
        tc = tc.unsqueeze(-1)

        dP = w + (sc * u) - (tc * v)
        return torch.sqrt(torch.sum(dP * dP, dim=-1) + 1e-8)

    # ------------------------------------------------------------------
    # 충돌 Loss
    # ------------------------------------------------------------------
    def get_collision_loss(self, global_pos, colliding_pairs):
        """FK로 복원한 관절 위치 딕셔너리에서 캡슐 충돌 침투량 기반 Loss를 계산."""
        loss = 0.0
        for (p1, c1), (p2, c2) in colliding_pairs:
            dist = self.capsule_distance(global_pos[p1], global_pos[c1],
                                         global_pos[p2], global_pos[c2])
            threshold = self.bone_radii[c1] + self.bone_radii[c2]
            penetration = torch.relu(threshold - dist) * 10.0
            loss = loss + torch.pow(penetration, 2).mean()
        return loss

    def get_collision_loss_from_quats(self, hips_pos, quats_84, colliding_pairs):
        """
        새 양식 진입점: (Hips 위치 + 84 쿼터니언)에서 FK를 수행하고 충돌 Loss를 반환.
        hips_pos: [..., 3], quats_84: [..., 84]
        """
        global_pos = self.forward_kinematics(hips_pos, quats_84)
        return self.get_collision_loss(global_pos, colliding_pairs)

    # ------------------------------------------------------------------
    # 해석 가능한 지표용: '선형' 침투 깊이 (m)
    # ------------------------------------------------------------------
    # get_collision_loss는 학습용(스케일 ×10 후 제곱)이라 물리 단위가 아니다.
    # 아래 함수들은 스케일/제곱 없이 relu(threshold - dist) 그대로의 깊이를 돌려주므로
    # cm 단위 보고, 충돌 프레임 비율, 깊이 기준 제거율 등 정직한 지표 계산에 사용한다.
    # evaluate.py / demo_maker.py / 시각화가 모두 이 함수를 공유해 지표가 어긋나지 않게 한다.

    def get_penetration_depths(self, global_pos, colliding_pairs):
        """
        FK로 복원된 관절 위치 딕셔너리에서 페어별 선형 침투 깊이(m)를 반환.
        반환: [..., n_pairs]  (침투 없으면 0)
        """
        depths = []
        for (p1, c1), (p2, c2) in colliding_pairs:
            dist = self.capsule_distance(global_pos[p1], global_pos[c1],
                                         global_pos[p2], global_pos[c2])
            threshold = self.bone_radii[c1] + self.bone_radii[c2]
            depths.append(torch.relu(threshold - dist))
        return torch.stack(depths, dim=-1)

    def get_penetration_depths_from_quats(self, hips_pos, quats_84, colliding_pairs):
        """(Hips 위치 + 84 쿼터니언)에서 FK 후 페어별 선형 침투 깊이(m) [..., n_pairs] 반환."""
        global_pos = self.forward_kinematics(hips_pos, quats_84)
        return self.get_penetration_depths(global_pos, colliding_pairs)
