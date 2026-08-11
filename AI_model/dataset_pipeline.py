"""
데이터 계약 + 학습용 데이터셋 + 실행 경로 헬퍼 (전 모듈 공용 코어).

이 파일은 train / evaluate / inference / demo_maker / corruption 이 공통으로 import하는
'계약'만 담는다. 무거운 의존성(matplotlib / scipy / pandas)을 두지 않는 것이 원칙이다.

  - 뼈대 계약 상수 : PARENTS / BONE_NAMES / BONE_MAP / BONE_RADII
  - 학습 데이터셋   : BandaiMotionDataset
  - 결정론적 분할   : get_split_files (VAL_RATIO / SPLIT_SEED)
  - 체크포인트 헬퍼 : make_run_name / resolve_ckpt_root / find_*_checkpoint*/run*

[2026-08-07 분리] 구판은 이 파일 하나가 MODE(PREPROCESS/VISUALIZE) ×
VISUALIZE_TYPE(SINGLE/COMPARE)로 3중 분기하는 257줄 __main__을 갖고 있었다. 이제:
  - CSV → .pt 전처리  →  preprocess.py
  - 3D 시각화          →  motion_viz.py   (single / compare 서브커맨드)
이동한 함수 7개는 전부 구 __main__에서만 호출되던 것이라 import 계약은 변하지 않았다.
부수 효과: train.py가 더 이상 matplotlib/scipy를 로드하지 않는다.
"""
import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

PARENTS = {
    'Hips': None, 'Spine': 'Hips', 'Chest': 'Spine', 'Neck': 'Chest', 'Head': 'Neck',
    'LeftShoulder': 'Chest', 'LeftUpperArm': 'LeftShoulder', 'LeftLowerArm': 'LeftUpperArm', 'LeftHand': 'LeftLowerArm',
    'RightShoulder': 'Chest', 'RightUpperArm': 'RightShoulder', 'RightLowerArm': 'RightUpperArm', 'RightHand': 'RightLowerArm',
    'LeftUpperLeg': 'Hips', 'LeftLowerLeg': 'LeftUpperLeg', 'LeftFoot': 'LeftLowerLeg', 'LeftToes': 'LeftFoot',
    'RightUpperLeg': 'Hips', 'RightLowerLeg': 'RightUpperLeg', 'RightFoot': 'RightLowerLeg', 'RightToes': 'RightFoot'
}

BONE_NAMES = sorted(list(PARENTS.keys()))
BONE_MAP = {name: i for i, name in enumerate(BONE_NAMES)}
BONE_RADII = {
    'Hips': 0.06, 'Spine': 0.04, 'Chest': 0.08, 'Neck': 0.03, 'Head': 0.05,
    'LeftShoulder': 0.03, 'LeftUpperArm': 0.03, 'LeftLowerArm': 0.02, 'LeftHand': 0.02,
    'RightShoulder': 0.03, 'RightUpperArm': 0.03, 'RightLowerArm': 0.02, 'RightHand': 0.02,
    'LeftUpperLeg': 0.05, 'LeftLowerLeg': 0.04, 'LeftFoot': 0.03, 'LeftToes': 0.02,
    'RightUpperLeg': 0.05, 'RightLowerLeg': 0.04, 'RightFoot': 0.03, 'RightToes': 0.02
} # 뼈대 Capsulize

# ============================================================
# Train / Test 분할 (과적합 검증용 held-out set)
# ============================================================
# 모든 스크립트(train / evaluate / inference)가 동일한 결정론적 분할을 공유하도록
# 고정 seed로 셔플한 뒤 앞쪽 VAL_RATIO 비율을 held-out(test)으로, 나머지를 train으로 사용.
VAL_RATIO = 0.1     # 전체의 10%를 학습에 쓰지 않고 평가 전용으로 격리
SPLIT_SEED = 42     # 분할 재현성을 위한 고정 시드

def get_split_files(pt_dir, split='train', val_ratio=VAL_RATIO, seed=SPLIT_SEED):
    """
    pt_dir 내 .pt 파일을 결정론적으로 train / test 로 나눈다.
      split='train' -> 학습용, 'test'(='val') -> 평가/추론용 held-out, 'all' -> 전체
    """
    files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))   # sorted로 순서 고정
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = int(len(files) * val_ratio)
    test_files = files[:n_val]
    train_files = files[n_val:]

    if split == 'train':
        return train_files
    if split in ('test', 'val'):
        return test_files
    return files


# ============================================================
# 체크포인트 폴더 관리 (실험용 가중치별로 폴더 분리)
# ============================================================
# 실험마다 checkpoints/ 아래에 사용된 손실 가중치를 이름으로 하는 하위 폴더를 만들어
# 가중치 파일 / run_config.json / log.txt 를 격리한다. 이렇게 하면 lambda 스윕 시
# 이전 실험 결과가 덮어써지지 않고, 폴더 이름만 봐도 어떤 설정인지 알 수 있다.

def make_run_name(lambda_recon, lambda_phys, beta_kl, tag=None):
    """
    실험 손실 가중치를 사람이 읽기 쉬운 폴더명으로 변환. 예: recon1_phys0.5_kl0.01
    tag를 주면 접두어로 붙는다 (예: tag='declip' → declip_recon1_phys0.5_kl0.01).
    → §4.1 디클리핑 시대의 run이 정규화 시대(무태그) 폴더와 절대 충돌하지 않게 하는 장치.
    """
    def fmt(v):
        return f"{float(v):g}"   # 1.0 -> '1', 0.5 -> '0.5', 0.01 -> '0.01'
    base = f"recon{fmt(lambda_recon)}_phys{fmt(lambda_phys)}_kl{fmt(beta_kl)}"
    return f"{tag}_{base}" if tag else base


def resolve_ckpt_root():
    """실행 위치(project root / AI_model)에 무관하게 checkpoints 루트 경로를 반환."""
    return "checkpoints" if os.path.exists("checkpoints") else "../checkpoints"


def find_latest_run_dir(ckpt_root=None):
    """
    가중치가 들어있는 run 하위 폴더 중 가장 최근에 학습된 폴더 경로를 반환한다.
    하위 폴더 구조가 없으면 구(舊) 평면 구조(ckpt_root 직속에 .pth) 호환으로 ckpt_root를 반환.
    아무 가중치도 없으면 None.
    """
    ckpt_root = ckpt_root or resolve_ckpt_root()
    run_dirs = [d for d in glob.glob(os.path.join(ckpt_root, "*"))
                if os.path.isdir(d) and glob.glob(os.path.join(d, "pvtvae_epoch_*.pth"))]
    if run_dirs:
        return max(run_dirs, key=os.path.getmtime)
    if glob.glob(os.path.join(ckpt_root, "pvtvae_epoch_*.pth")):
        return ckpt_root   # 구 평면 구조 호환
    return None


def find_latest_checkpoint_in(run_dir):
    """주어진 run 폴더에서 가장 마지막 epoch 가중치 경로를 반환 (없으면 None)."""
    ckpts = glob.glob(os.path.join(run_dir, "pvtvae_epoch_*.pth"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))


def find_run_dir_by_config(lambda_recon, lambda_phys, beta_kl, tag=None, ckpt_root=None):
    """
    손실 가중치 조합으로 특정 실험의 run 폴더를 직접 찾는다 (mtime이 아니라 설정으로 선택).
    train.py에서 쓰던 값과 동일한 (recon, phys, kl[, tag])을 주면 그 실험의 폴더를 반환하므로,
    가장 최근 실험이 아니라 '과거의 특정 실험'도 다시 평가할 수 있다.
    (tag=None이면 구(舊) 정규화-시대 폴더를, tag='declip'이면 디클리핑-시대 폴더를 찾는다.)
    가중치(.pth)가 들어있는 해당 폴더 경로를 반환하고, 없으면 None.
    """
    ckpt_root = ckpt_root or resolve_ckpt_root()
    run_dir = os.path.join(ckpt_root, make_run_name(lambda_recon, lambda_phys, beta_kl, tag=tag))
    if os.path.isdir(run_dir) and glob.glob(os.path.join(run_dir, "pvtvae_epoch_*.pth")):
        return run_dir
    return None


def list_available_runs(ckpt_root=None):
    """가중치가 들어있는 모든 run 폴더 이름 목록 (에러 메시지에서 사용자 안내용)."""
    ckpt_root = ckpt_root or resolve_ckpt_root()
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(ckpt_root, "*"))
                  if os.path.isdir(d) and glob.glob(os.path.join(d, "pvtvae_epoch_*.pth")))


# [수정] 머신러닝 데이터셋 클래스 (훈련 시에는 84차원 쿼터니언만 분리 제공)
class BandaiMotionDataset(Dataset):
    def __init__(self, pt_dir, seq_len=30, split='train'):
        # split='train'은 held-out test 파일을 제외한 학습용 파일만 로드
        self.pt_files = get_split_files(pt_dir, split=split)
        self.seq_len = seq_len
        self.data = []

        for f in self.pt_files:
            tensor = torch.load(f) # [Frames, 87]
            if tensor.shape[0] >= seq_len:
                self.data.append(tensor)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        tensor = self.data[idx]
        start = np.random.randint(0, tensor.shape[0] - self.seq_len + 1)
        window = tensor[start : start + self.seq_len] # [30, 87]

        # 새 양식: Hips Position(3) + 21 Quaternion(84) = 87 전체를 모델 입력으로 반환
        return window


# ============================================================
# 실행 호환 shim
# ============================================================
# 이 파일은 더 이상 자체 뷰어/전처리를 갖지 않는다 (2026-08-07 분리).
# 다만 기존 워크플로(`python AI_model/dataset_pipeline.py`, VS Code의 "Run Python File")가
# 그대로 동작하도록 시각화로 위임한다 — 구판 기본값이 VISUALIZE/COMPARE였으므로 동일하다.
if __name__ == "__main__":
    print("ℹ️ dataset_pipeline.py는 데이터 계약 모듈이 되었습니다 "
          "(시각화 → motion_viz.py, 전처리 → preprocess.py).")
    print("   구 기본 동작(COMPARE)으로 motion_viz.py에 위임합니다. "
          "다른 모드는 'python AI_model/motion_viz.py single' 등으로 실행하세요.\n")
    # import는 이 블록 안에서만 — 모듈로 import될 때는 matplotlib/scipy가 로드되지 않는다.
    import sys

    import motion_viz
    motion_viz.main(sys.argv[1:])
