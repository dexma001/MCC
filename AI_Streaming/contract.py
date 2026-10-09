"""입출력 계약 — 라이브 입력이 학습 분포와 '같은 것'임을 보장하는 유일한 지점.

[이 파일이 존재하는 이유]
  `claude_analysis/vtuber_live_deployment_gap_20260911.md` §3.1 의 P2-1:
  87차원 텐서의 규약(관절 순서·쿼터니언 성분 순서·단위)이 틀려도 **예외가 나지 않는다.**
  그럴듯한 모양의 텐서가 그대로 흘러가 포즈만 조용히 망가진다. 학습 파이프라인에는
  이 규약을 검사하는 코드가 한 줄도 없다 (검사할 필요가 없었다 — 입력이 항상 자기 전처리
  산출물이었으므로). 라이브에서는 입력이 남의 프로그램에서 오므로 검사가 필수다.

[87차원의 정확한 정의 — AI_model/preprocess.py 에서 역산]
    [0:3]   Hips 월드 좌표, 단위 **m** (원본 CSV 는 cm, 전처리에서 /100)
    [3:87]  21관절 로컬 쿼터니언, 관절당 **(x, y, z, w)** — w 가 마지막
    관절 순서 = sorted(PARENTS.keys()) = **알파벳순** (Chest, Head, Hips, LeftFoot, ...)
    좌표계 = Unity Y-Up, 회전은 **부모 기준 로컬**
    프레임률 = 30fps, 윈도우 = 30프레임(정확히 1초)

[Hips 단위에 대한 실측 메모]
  cm/m 를 혼동해도 관통 판정은 영향 0 이다 — 모든 충돌 페어가 상대 거리라 루트가 상쇄된다
  (실측 포즈 차이 0.057cm). 그래도 여기서 m 로 못박는다: FK 로 복원한 신장으로 계약을
  검사하는 것이 가장 값싼 오배선 탐지기이기 때문이다.
"""

import paths  # noqa: F401  (sys.path 부트스트랩)

import torch

from dataset_pipeline import BONE_NAMES, PARENTS, BONE_RADII, SKELETON_STATURE_M

# =====================================================================
# 계약 상수
# =====================================================================
FRAME_DIM = 87
NUM_JOINTS = 21
SEQ_LEN = 30
"""모델의 위치 임베딩 길이. 31 이상은 RuntimeError, 29 이하는 '조용히 다른 답'."""
FPS = 30
QUAT_ORDER = "xyzw"
HIPS_UNIT = "m"

BONE_ORDER = tuple(BONE_NAMES)
"""관절 순서. dataset_pipeline 의 정의를 그대로 쓴다 (복사하지 않는다)."""

BONE_INDEX = {name: i for i, name in enumerate(BONE_ORDER)}

COLLIDING_PAIRS = [
    (('Hips', 'Chest'), ('LeftLowerArm', 'LeftHand')),
    (('Hips', 'Chest'), ('RightLowerArm', 'RightHand')),
    (('LeftLowerArm', 'LeftHand'), ('RightLowerArm', 'RightHand')),
    (('LeftLowerLeg', 'LeftFoot'), ('RightLowerLeg', 'RightFoot')),
]
"""사영이 보장하는 4쌍 = 학습이 실제로 최적화한 페어.

**배포용 사본**이다 — 런타임이 train.py(=corruption.py 39KB + 학습 경로)를 끌어오지 않도록
값을 여기 둔다. 원본과 갈라지면 보장 대상이 달라지므로 `check_training_contract()` 가 검사한다.
"""


UNITY_HUMAN_BODY_BONES = frozenset([
    "Hips", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg", "LeftFoot", "RightFoot",
    "Spine", "Chest", "UpperChest", "Neck", "Head", "LeftShoulder", "RightShoulder",
    "LeftUpperArm", "RightUpperArm", "LeftLowerArm", "RightLowerArm", "LeftHand", "RightHand",
    "LeftToes", "RightToes", "LeftEye", "RightEye", "Jaw",
    "LeftThumbProximal", "LeftThumbIntermediate", "LeftThumbDistal",
    "LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
    "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
    "LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
    "LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",
    "RightThumbProximal", "RightThumbIntermediate", "RightThumbDistal",
    "RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal",
    "RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal",
    "RightRingProximal", "RightRingIntermediate", "RightRingDistal",
    "RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal",
])
"""Unity HumanBodyBones 이름 전체 (VMC /VMC/Ext/Bone/Pos 의 name 인자가 이 철자여야 한다).
우리 21본이 전부 이 집합에 있어야 Warudo 가 받는다 — check_vmc_names() 가 검사한다.
UpperChest 는 우리 리그에 없다(트래커가 보내면 vmc_bridge 가 Chest 에 접는다)."""


def check_vmc_names():
    """21본 이름이 전부 Unity HumanBodyBones 철자인지. 문제 리스트 반환."""
    bad = [b for b in BONE_ORDER if b not in UNITY_HUMAN_BODY_BONES]
    return [f"Unity HumanBodyBones 에 없는 본 이름: {bad}"] if bad else []


# =====================================================================
# 변환
# =====================================================================
def pack_frame(hips_xyz_m, quat_by_bone, dtype=torch.float32):
    """(Hips 위치[m], {본이름: (x,y,z,w)}) -> [87] 텐서.

    라이브 수신부(OSC 등)가 만든 딕셔너리를 모델 입력 규약으로 바꾸는 **유일한 경로**다.
    본이 하나라도 빠지면 KeyError 로 즉시 실패한다 — 조용히 0 으로 채우지 않는다.
    """
    hips = torch.as_tensor(hips_xyz_m, dtype=dtype).reshape(3)
    quats = torch.empty(NUM_JOINTS, 4, dtype=dtype)
    for i, name in enumerate(BONE_ORDER):
        q = torch.as_tensor(quat_by_bone[name], dtype=dtype).reshape(4)
        quats[i] = q
    return torch.cat([hips, quats.reshape(-1)])


def unpack_frame(frame87):
    """[87] -> (Hips 위치[3], {본이름: [4] (x,y,z,w)})."""
    f = frame87.reshape(FRAME_DIM)
    hips = f[:3]
    quats = f[3:].reshape(NUM_JOINTS, 4)
    return hips, {name: quats[i] for i, name in enumerate(BONE_ORDER)}


def normalize_quats(frame87, eps=1e-8):
    """관절별 단위 쿼터니언으로 재정규화한 [87] 사본. 원본은 건드리지 않는다."""
    f = frame87.clone().reshape(FRAME_DIM)
    q = f[3:].reshape(NUM_JOINTS, 4)
    f[3:] = (q / (q.norm(dim=-1, keepdim=True) + eps)).reshape(-1)
    return f


# =====================================================================
# 검사
# =====================================================================
def validate_frame(frame87, quat_tol=1e-3, stature_tol=0.35, physics=None):
    """프레임 하나가 계약을 지키는지 검사. 문제 문자열 리스트를 돌려준다(빈 리스트 = 정상).

    검사 항목 (전부 '조용히 틀리는' 오류를 겨냥한다):
      1. 모양/NaN  — 배선 사고
      2. 쿼터니언 노름 ≈ 1 — 성분 순서가 틀리면 대개 여기서 먼저 걸린다
      3. FK 로 복원한 신장이 리그 신장(1.508m) 근처인가 — **관절 순서/단위 오배선 탐지기**
         (physics 를 넘겼을 때만. 순서가 뒤섞이면 뼈가 엉뚱한 방향으로 붙어 신장이 무너진다)
    """
    problems = []
    f = torch.as_tensor(frame87)
    if f.numel() != FRAME_DIM:
        return [f"차원이 {f.numel()} 입니다 (기대 {FRAME_DIM})"]
    f = f.reshape(FRAME_DIM)
    if not torch.isfinite(f).all():
        problems.append("NaN/Inf 가 포함돼 있습니다")
        return problems

    q = f[3:].reshape(NUM_JOINTS, 4)
    norm_err = (q.norm(dim=-1) - 1.0).abs().max().item()
    if norm_err > quat_tol:
        problems.append(f"쿼터니언 노름 오차 {norm_err:.4f} > {quat_tol} "
                        f"(성분 순서가 (x,y,z,w) 가 아닐 수 있습니다)")

    if physics is not None:
        gp = physics.forward_kinematics(f[:3].reshape(1, 3), f[3:].reshape(1, 84))
        head = gp['Head'].reshape(3)
        foot = torch.minimum(gp['LeftFoot'].reshape(3), gp['RightFoot'].reshape(3))
        stature = float((head - foot).norm())
        if abs(stature - SKELETON_STATURE_M) > stature_tol:
            problems.append(
                f"FK 복원 신장 {stature:.3f}m 가 리그 신장 {SKELETON_STATURE_M}m 와 "
                f"{stature_tol}m 이상 차이납니다 (관절 순서나 단위 오배선 의심)")
    return problems


def check_training_contract():
    """학습 코드와 계약이 갈라지지 않았는지 검사. (문제 리스트, 상세 dict) 반환.

    import 는 이 함수 안에서만 한다 — 런타임이 train.py 를 끌어오지 않게 하려는 것이다.
    """
    import importlib.util
    import os

    problems = []
    detail = {}

    spec = importlib.util.spec_from_file_location(
        "_ai_model_train_for_check", os.path.join(paths.AI_MODEL_DIR, "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if list(mod.COLLIDING_PAIRS) != list(COLLIDING_PAIRS):
        problems.append("COLLIDING_PAIRS 가 train.py 와 다릅니다 (사영 보장 대상이 달라집니다)")
    detail["pairs"] = list(COLLIDING_PAIRS)

    if tuple(BONE_NAMES) != BONE_ORDER:
        problems.append("BONE_ORDER 가 dataset_pipeline.BONE_NAMES 와 다릅니다")
    detail["n_bones"] = len(BONE_ORDER)
    detail["bone_order_head"] = BONE_ORDER[:4]

    if len(PARENTS) != NUM_JOINTS:
        problems.append(f"관절 수가 {len(PARENTS)} 입니다 (기대 {NUM_JOINTS})")

    missing = [b for b in BONE_ORDER if b not in BONE_RADII]
    if missing:
        problems.append(f"BONE_RADII 에 없는 본: {missing}")
    detail["radii_stature_m"] = SKELETON_STATURE_M
    return problems, detail


def make_physics(device="cpu"):
    """배포용 물리 엔진 (FK + 캡슐 거리). 사영과 검사에 모두 이 인스턴스를 쓴다."""
    from physics_module import DifferentiablePhysics
    return DifferentiablePhysics(PARENTS, BONE_RADII, offset_csv_path=paths.OFFSET_CSV).to(device)
