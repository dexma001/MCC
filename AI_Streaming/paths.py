"""경로 부트스트랩 — AI_Streaming 의 모든 모듈이 가장 먼저 import 한다.

[하는 일]
  1. `AI_model/` 을 sys.path 에 얹는다 (원본을 import 하기 위해. **수정하지 않는다**).
  2. 프로젝트 루트 기준의 경로 헬퍼를 제공한다. AI_model 의 모듈들은 `checkpoints`,
     `processed_motions_VMC`, `Sample_Data/...` 를 **cwd 상대 경로**로 찾으므로,
     스트리밍 프로세스를 어디서 실행하든 같은 파일을 보게 하려면 절대 경로가 필요하다.

[이름 충돌 방지]
  이 폴더의 모듈 이름(paths / contract / corrector / sources / sinks / run_sim / selftest)은
  AI_model 의 어떤 파일과도 겹치지 않는다. 따라서 sys.path 순서가 결과를 바꾸지 않는다.
  (AI_model_large 에서 models.py/train.py 파리티를 시도했다가 evaluate.py 의 내부 import 가
   깨졌던 것과 같은 함정을 처음부터 피한 것이다.)
"""

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AI_MODEL_DIR = os.path.join(ROOT, "AI_model")
CHECKPOINTS_DIR = os.path.join(ROOT, "checkpoints")
MOTIONS_DIR = os.path.join(ROOT, "processed_motions_VMC")
OFFSET_CSV = os.path.join(ROOT, "Sample_Data", "Standard_BoneOffsets.csv")

if AI_MODEL_DIR not in sys.path:
    sys.path.insert(0, AI_MODEL_DIR)

DEFAULT_RUN = "tfm_declip_cov_l10.1_anat_recon1_phys0.5_kl0"
"""배포 기준선 = 동결 v1 (anat 반지름, λ_phys=0.5).

배포 구성(모델 -> 저역통과 -> 사영)으로 평가된 이력이 가장 길고, 스트리밍 위험 분석
(2026-09-04)도 이 체크포인트로 측정됐다. 다른 런을 쓰려면 --run 으로 지정한다.
"""


def checkpoint_path(run_name=None, epoch=None):
    """`checkpoints/<run_name>/pvtvae_epoch_<epoch>.pth` 의 절대 경로.

    epoch 를 주지 않으면 폴더에서 가장 큰 epoch 를 고른다.
    파일명 규칙 'pvtvae_epoch_*.pth' 는 프로젝트 전역 계약이다 (이름을 바꾸면 공유 헬퍼가
    전부 깨진다) — 여기서도 같은 규칙으로 찾는다.
    """
    run_name = run_name or DEFAULT_RUN
    run_dir = run_name if os.path.isdir(run_name) else os.path.join(CHECKPOINTS_DIR, run_name)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(
            f"체크포인트 폴더를 찾을 수 없습니다: {run_dir}\n"
            f"  사용 가능: {available_runs()}")
    if epoch is not None:
        p = os.path.join(run_dir, f"pvtvae_epoch_{epoch}.pth")
        if not os.path.exists(p):
            raise FileNotFoundError(f"해당 epoch 가중치가 없습니다: {p}")
        return p
    ckpts = glob.glob(os.path.join(run_dir, "pvtvae_epoch_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"폴더에 pvtvae_epoch_*.pth 가 없습니다: {run_dir}")
    return max(ckpts, key=lambda x: int(re.search(r"pvtvae_epoch_(\d+)\.pth", x).group(1)))


def available_runs():
    """가중치가 들어있는 run 폴더 이름 목록."""
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(CHECKPOINTS_DIR, "*"))
                  if os.path.isdir(d) and glob.glob(os.path.join(d, "pvtvae_epoch_*.pth")))


def test_motion_files(limit=0):
    """held-out(test) 스플릿의 .pt 파일 절대 경로 목록.

    스플릿 규칙(SPLIT_SEED=42, VAL_RATIO=0.1)은 dataset_pipeline 의 것을 그대로 쓴다 —
    학습에 쓴 파일로 '라이브 시뮬레이션'을 하면 결과가 낙관적으로 오염된다.
    """
    from dataset_pipeline import get_split_files
    files = get_split_files(MOTIONS_DIR, split='test')
    files = [os.path.abspath(f) for f in files]
    return files[:limit] if limit else files
