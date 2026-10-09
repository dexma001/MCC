"""리그 T-포즈(G0) 도출 — retarget.py 가 쓰는 rig_tpose.json 을 만든다.

    python AI_Streaming/derive_rest_pose.py            # 도출 + 검증 + 저장
    python AI_Streaming/derive_rest_pose.py --check    # 저장하지 않고 검증만

[문제] 이 리그에는 T-포즈가 어디에도 정의돼 있지 않다. 원본 CSV 는 FBX 리그의 로컬 회전
  (관절 방향 내장)이고, 데이터셋 어느 프레임도 T-포즈가 아니며, Unity 휴머노이드 설정을
  거친 흔적도 없다. 그래서 Warudo 가 요구하는 "T-포즈 = 회전 0" 기준을 **기하로 구성**한다.

[방법] 각 본 b 의 전역 회전 G0_b(T-포즈)를 다음 두 조건으로 정한다.
  (1) 방향: 본의 로컬 +Y(자식 오프셋 방향)가 T-포즈 세그먼트 방향 d0 를 가리킨다.
        척추 사슬 +Y, 팔 ±X(수평), 다리 -Y, 발은 '서 있을 때의 평균 방향'
        (발은 아바타의 자체 발 기울기를 쓰는 것이 옳으므로 데이터의 서 있는 방향을 T-포즈로 삼는다).
        잎 본(Head/Hand/Toes)은 자식이 없어 방향 조건이 없다 — 부모의 스윙을 상속한다.
  (2) 비틀림: Hips 는 다리 오프셋(±X), Chest 는 어깨 오프셋(±Z→∓X)이 좌우를 가리키도록 고정.
        나머지 본은 데이터셋 전체의 (yaw 정규화된) 평균 전역 회전에서 d0 로 가는 **최소 회전**만
        적용해 비틀림을 보존한다 — 팔을 내린 A-포즈에서 T-포즈로 올릴 때 팔이 비틀리지 않는다는
        통상의 리타깃 가정과 같다.
  평균은 '서 있는' 프레임(Hips→Head 가 +Y 와 25° 이내)만 쓰고, 프레임마다 다리 위치로 잰
  앞 방향을 +Z 로 돌린 뒤(yaw 정규화) 합친다. 평균은 Σ qqᵀ 의 최대 고유벡터로 구한다
  (q 와 -q 를 같은 회전으로 보는 부호 무관 평균 — 단순 합은 180° 근처에서 무너진다).

[검증 — 이 스크립트가 매번 출력한다]
  A. 도출된 T-포즈를 리그 FK 로 그리면 팔이 수평 ±X, 다리 -Y, 머리 +Y 인가
  B. 좌우 프레임 규약 (정보만 — 이 리그는 오른쪽 프레임이 거울이 아니라 180° 뒤집힘)
  C. 위치 보존: VRM 쪽 오프셋으로 FK 한 위치가 리그 FK 와 일치 (대수적 항등 — 구현 오류 검출)
  D. 왕복 vrm_to_rig(rig_to_vrm(x)) == x
  E. 데이터셋에서 VRM 로컬 회전의 크기 분포 (Hips/척추 작고, UpperArm ≈ 팔 내린 각도)
  F. 좌우 대칭 (VRM 로컬 크기의 Left/Right 중앙값 차)
  G. 잎 본: 서 있는 자세에서 Head 가 단위 회전 근처인가
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import paths

import torch

import contract
import retarget as rt
from dataset_pipeline import PARENTS

X, Y, Z = rt._X, rt._Y, rt._Z

# 본의 로컬 +Y 가 T-포즈에서 가리키는 월드 방향. 'mean' = 서 있는 자세의 평균 방향을 그대로.
TPOSE_DIR = {
    'Hips': Y, 'Spine': Y, 'Chest': Y, 'Neck': Y, 'Head': 'inherit',
    'LeftShoulder': -X, 'LeftUpperArm': -X, 'LeftLowerArm': -X, 'LeftHand': 'inherit',
    'RightShoulder': X, 'RightUpperArm': X, 'RightLowerArm': X, 'RightHand': 'inherit',
    'LeftUpperLeg': -Y, 'LeftLowerLeg': -Y, 'LeftFoot': 'mean', 'LeftToes': 'inherit',
    'RightUpperLeg': -Y, 'RightLowerLeg': -Y, 'RightFoot': 'mean', 'RightToes': 'inherit',
}
# 'inherit' = 잎 본(자식 없음). 잎의 로컬 축은 FBX 가 임의로 정한 것이라(예: Head 의 +Y 가 앞을
#   향할 수 있다) 방향 조건을 걸 수 없다. 대신 **부모의 스윙을 그대로 상속**한다 — 손은 전완과
#   함께 올라가고, 머리는 목과 함께, 발끝은 발과 함께 움직인 것이 T-포즈다.
#   (처음 구현에서 잎에도 방향 조건을 걸었더니 Head 의 VRM 회전이 평균 94° 로 나왔다.)
# 비틀림을 기하로 고정하는 본: (측면 자식, 그 자식이 T-포즈에서 놓일 월드 방향)
LATERAL_FIX = {'Hips': ('LeftUpperLeg', -X), 'Chest': ('LeftShoulder', -X)}

UPRIGHT_COS = math.cos(math.radians(25.0))


def _iter_batches(file_stride=3, frame_stride=5, max_files=0):
    files = sorted(glob.glob(os.path.join(paths.MOTIONS_DIR, "*.pt")))[::file_stride]
    if max_files:
        files = files[:max_files]
    for f in files:
        m = torch.load(f)[::frame_stride]
        if m.shape[0] == 0:
            continue
        yield f, m


def _facing_yaw_fix(P):
    """프레임별로 앞 방향(다리 위치 기준)을 +Z 로 돌리는 월드 회전 [N,4]."""
    left = P['LeftUpperLeg'] - P['RightUpperLeg']
    fwd = torch.cross(Y.expand_as(left), left, dim=-1)          # 위 × 왼쪽 = 앞
    fwd_flat = fwd * torch.tensor([1.0, 0.0, 1.0])
    return rt.q_swing(fwd_flat, Z.expand_as(fwd_flat))


def derive(physics, verbose=True, max_files=0):
    bi = contract.BONE_INDEX
    M = torch.zeros(contract.NUM_JOINTS, 4, 4, dtype=torch.float64)   # Σ q qᵀ (본별)
    n_used = n_seen = 0
    t0 = time.time()
    for f, m in _iter_batches(max_files=max_files):
        q = m[:, 3:].reshape(-1, contract.NUM_JOINTS, 4)
        P, G = rt.fk_with_offsets(m[:, :3], q, physics.bone_offsets)
        n_seen += m.shape[0]
        up = P['Head'] - P['Hips']
        upright = (up / (up.norm(dim=-1, keepdim=True) + 1e-9))[:, 1] > UPRIGHT_COS
        if not upright.any():
            continue
        R = _facing_yaw_fix({k: v[upright] for k, v in P.items()})
        Gs = torch.stack([rt.qmul(R, G[b][upright]) for b in contract.BONE_ORDER], dim=1)  # [n,21,4]
        Gs = Gs.double()
        M += torch.einsum('nbi,nbj->bij', Gs, Gs)
        n_used += int(upright.sum())
    evals, evecs = torch.linalg.eigh(M)                    # 오름차순
    Gm = rt.qnorm(evecs[:, :, -1].float())                 # 최대 고유벡터 = 부호 무관 평균 회전
    if verbose:
        print(f"  평균 전역 회전: {n_used}/{n_seen} 프레임(서 있는 것만), {time.time()-t0:.1f}s")

    # (1) 방향 맞추기: 본 ŷ 의 평균 방향 -> d0
    G0 = torch.zeros_like(Gm)
    d0_used = {}
    swing = {}
    for b in rt._ORDER:                                    # 부모가 먼저 (상속 때문에 순서 중요)
        i = bi[b]
        d_mean = rt.qrot(Gm[i], Y)
        spec = TPOSE_DIR[b]
        if isinstance(spec, str) and spec == 'inherit':
            swing[b] = swing[PARENTS[b]]
            d0 = rt.qrot(swing[b], d_mean)
        else:
            d0 = d_mean if isinstance(spec, str) else spec
            d0 = d0 / d0.norm()
            swing[b] = rt.q_swing(d_mean, d0)
        G0[i] = rt.qmul(swing[b], Gm[i])
        d0_used[b] = d0
    # (2) 비틀림 고정 (Hips / Chest)
    for b, (child, target) in LATERAL_FIX.items():
        i = bi[b]
        axis = d0_used[b]
        w = rt.qrot(G0[i], physics.bone_offsets[child])
        w = w - (w @ axis) * axis
        t = target - (target @ axis) * axis
        w, t = w / w.norm(), t / t.norm()
        ang = torch.atan2((torch.cross(w, t, dim=-1) @ axis), (w @ t))
        G0[i] = rt.qmul(rt.q_axis_angle(axis, ang), G0[i])
    G0 = rt.qnorm(G0)
    return G0, Gm, d0_used, n_used


def validate(G0, physics, verbose=True):
    """도출 결과 검증. 실패 문자열 리스트 반환 (빈 리스트 = 통과)."""
    bi = contract.BONE_INDEX
    problems = []
    G0p = torch.stack([G0[bi[PARENTS[b]]] if PARENTS[b] else rt.QID for b in contract.BONE_ORDER])
    L0 = rt.qmul(rt.qconj(G0p), G0)

    # A. T-포즈 기하
    hips0 = torch.zeros(1, 3)
    P, _ = rt.fk_with_offsets(hips0, L0.unsqueeze(0), physics.bone_offsets)
    P = {k: v[0] for k, v in P.items()}
    if verbose:
        print("  [A] 리그 T-포즈 FK (m):")
        for b in ['Head', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'LeftToes', 'LeftShoulder']:
            print(f"      {b:14s} {[round(x, 3) for x in P[b].tolist()]}")

    def seg(a, b):
        v = P[b] - P[a]
        return v / v.norm()
    checks = [
        ('왼팔 수평(-X)', seg('LeftShoulder', 'LeftHand'), -X),
        ('오른팔 수평(+X)', seg('RightShoulder', 'RightHand'), X),
        ('왼다리(-Y)', seg('LeftUpperLeg', 'LeftFoot'), -Y),
        ('오른다리(-Y)', seg('RightUpperLeg', 'RightFoot'), -Y),
        ('척추(+Y)', seg('Hips', 'Head'), Y),
        ('왼어깨는 왼쪽(-X)', seg('Chest', 'LeftShoulder') * torch.tensor([1.0, 0, 1.0]), -X),
    ]
    for name, v, want in checks:
        v = v / (v.norm() + 1e-9)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(v @ want)))))
        if verbose:
            print(f"      {name:18s} 편차 {ang:6.2f}°")
        if ang > 2.0:
            problems.append(f"T-포즈 기하: {name} 편차 {ang:.2f}°")
    toes = P['LeftToes'] - P['LeftFoot']
    foot_pitch = math.degrees(math.atan2(-float(toes[1]), float(toes[2])))
    if verbose:
        print(f"      발 기울기(발목→발끝, 수평 아래) {foot_pitch:.1f}°  "
              f"← 아바타의 발 기울기와 다르면 그 차이만큼 발 pitch 가 어긋난다")
    if not (0.0 <= foot_pitch <= 70.0):
        problems.append(f"발 방향이 이상합니다: {foot_pitch:.1f}°")

    # B. 좌우 프레임 규약 (정보) — 이 리그는 오른쪽 관절 프레임이 왼쪽의 거울이 아니라
    #    뼈 축 기준 180° 뒤집힌 규약이라 G0 자체는 거울 대칭이 아니다. 실패 조건이 아니라 기록.
    #    실제 대칭성은 validate_with_data 의 [F](VRM 로컬 크기 좌우 비교)로 판정한다.
    if verbose:
        print("  [B] 리그 좌우 프레임 규약: mirror(G0_Left) 대비 G0_Right 의 뼈축 비틀림차 (정보)")
        for lb in [b for b in contract.BONE_ORDER if b.startswith('Left')]:
            rb = 'Right' + lb[4:]
            d = rt.qmul(rt.qconj(G0[bi[rb]]), rt.q_mirror_x(G0[bi[lb]]))
            ang = float(rt.qangle_deg(d))
            print(f"      {lb:14s} {min(ang, 360.0 - ang):6.1f}°")
    return problems, L0


def validate_with_data(retg, physics, n_files=48, verbose=True):
    """C. 위치 보존  D. 왕복  E. VRM 로컬 각도 분포 — 실제 held-out 데이터로."""
    problems = []
    files = paths.test_motion_files(limit=n_files)
    vrm_off = retg.vrm_offsets(physics)
    worst_pos = worst_rt = 0.0
    angs = []
    for f in files:
        m = torch.load(f)[::7]
        v = retg.rig_to_vrm(m)
        back = retg.vrm_to_rig(v)
        d = (back[:, 3:].reshape(-1, 21, 4) * m[:, 3:].reshape(-1, 21, 4)).sum(-1).abs()
        worst_rt = max(worst_rt, float((1 - d).max()))
        P_rig = physics.compute_global_pos_tensor(m[:, :3], m[:, 3:])
        P_vrm, _ = rt.fk_with_offsets(v[:, :3], v[:, 3:].reshape(-1, 21, 4), vrm_off)
        P_vrm = torch.stack([P_vrm[b] for b in contract.BONE_ORDER], dim=1)
        worst_pos = max(worst_pos, float((P_rig - P_vrm).norm(dim=-1).max() * 100))
        angs.append(rt.qangle_deg(v[:, 3:].reshape(-1, 21, 4)))
    angs = torch.cat(angs)
    if verbose:
        print(f"  [C] 위치 보존: 리그 FK vs VRM 오프셋 FK 최대 차이 {worst_pos:.4f} cm")
        print(f"  [D] 왕복 rig->vrm->rig 최대 |1-|q·q'|| = {worst_rt:.2e}")
        print(f"  [E] VRM 로컬 회전 크기 (held-out {len(files)}파일, deg)  중앙값 / p95:")
        for b in contract.BONE_ORDER:
            a = angs[:, contract.BONE_INDEX[b]]
            a = torch.minimum(a, 360 - a)
            print(f"      {b:14s} {float(a.median()):6.1f} / {float(a.quantile(0.95)):6.1f}")
        print("  [F] 좌우 대칭 (VRM 로컬 크기 중앙값, Left vs Right):")
    #   평균이 아니라 중앙값이다 — 달리기 파일 몇 개가 한쪽 발을 크게 들어 6파일 평균에서는
    #   RightFoot 91° vs LeftFoot 27° 로 보였다(표본 편향). 770파일 중앙값은 22° vs 21° 였다.
    worst_sym = 0.0
    for lb in [b for b in contract.BONE_ORDER if b.startswith('Left')]:
        rb = 'Right' + lb[4:]
        al = torch.minimum(angs[:, contract.BONE_INDEX[lb]], 360 - angs[:, contract.BONE_INDEX[lb]]).median()
        ar = torch.minimum(angs[:, contract.BONE_INDEX[rb]], 360 - angs[:, contract.BONE_INDEX[rb]]).median()
        diff = abs(float(al - ar))
        worst_sym = max(worst_sym, diff)
        if verbose:
            print(f"      {lb[4:]:12s} L {float(al):6.1f}  R {float(ar):6.1f}  차 {diff:5.1f}°")
    # G. 서 있는 자세에서 머리·발이 T-포즈(단위) 근처인가 — 잎 본 축 처리의 검증
    if verbose:
        print("  [G] 잎 본 검증: Head/Toes 의 VRM 로컬 중앙값 (서 있는 자세 ≈ T-포즈여야 작다)")
        for b in ['Head', 'LeftToes', 'RightToes', 'Neck', 'Spine']:
            a = angs[:, contract.BONE_INDEX[b]]
            a = torch.minimum(a, 360 - a)
            print(f"      {b:10s} 중앙값 {float(a.median()):6.1f}°")
    head = angs[:, contract.BONE_INDEX['Head']]
    head = float(torch.minimum(head, 360 - head).median())
    if head > 30.0:
        problems.append(f"Head 의 VRM 로컬 회전 중앙값 {head:.1f}° — 잎 본 T-포즈가 틀렸습니다")
    if worst_sym > 15.0:
        problems.append(f"좌우 비대칭 {worst_sym:.1f}° (VRM 로컬 크기 중앙값의 좌우 차)")
    if worst_pos > 0.05:
        problems.append(f"위치 보존 실패 {worst_pos:.3f} cm (변환 구현 오류)")
    if worst_rt > 1e-5:
        problems.append(f"왕복 불일치 {worst_rt:.2e}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="저장하지 않고 검증만")
    ap.add_argument("--max-files", type=int, default=0, help="빠른 실험용 파일 수 제한")
    ap.add_argument("--out", default=rt.TPOSE_JSON)
    args = ap.parse_args(argv)

    physics = contract.make_physics("cpu")
    print("=" * 74)
    print("리그 T-포즈 도출")
    print("=" * 74)
    G0, Gm, d0, n_used = derive(physics, max_files=args.max_files)
    problems, L0 = validate(G0, physics)

    meta = {
        "_doc": "리그 T-포즈 전역 회전 G0[b] (x,y,z,w). derive_rest_pose.py 가 생성. "
                "VRM 로컬 = G0[parent]·L·inv(G0[b]).  좌표: Unity Y-Up, 앞 +Z, 왼쪽 -X.",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frames_used": n_used,
        "G0": {b: [round(float(x), 7) for x in G0[contract.BONE_INDEX[b]]] for b in contract.BONE_ORDER},
        "tpose_dir": {b: [round(float(x), 4) for x in d0[b]] for b in contract.BONE_ORDER},
    }
    tmp_path = args.out if not args.check else args.out + ".check.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    try:
        retg = rt.Retarget(tmp_path)
        problems += validate_with_data(retg, physics)
    finally:
        if args.check:
            os.remove(tmp_path)

    print("=" * 74)
    if problems:
        print("❌ 검증 실패:")
        for p in problems:
            print("   -", p)
        if not args.check:
            print(f"   (그래도 {args.out} 는 저장했습니다 — 수치를 보고 판단하세요)")
        return 1
    saved = "" if args.check else f" — 저장: {os.path.relpath(args.out, paths.ROOT)}"
    print(f"✅ 검증 통과{saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
