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

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AI_MODEL_DIR = os.path.join(ROOT, "AI_model")
CHECKPOINTS_DIR = os.path.join(ROOT, "checkpoints")
MOTIONS_DIR = os.path.join(ROOT, "processed_motions_VMC")
OFFSET_CSV = os.path.join(ROOT, "Sample_Data", "Standard_BoneOffsets.csv")

if AI_MODEL_DIR not in sys.path:
    sys.path.insert(0, AI_MODEL_DIR)

from dataset_pipeline import CHECKPOINT_FILENAME  # noqa: E402  ("temp.pth" — 위 sys.path 이후에만 import 가능)

CHECKPOINT = os.path.join(CHECKPOINTS_DIR, CHECKPOINT_FILENAME)
"""배포 가중치 = `checkpoints/temp.pth` 하나 (하위 폴더 없음).

개발판 tfm_declip_cov_l10.1_anat_recon1_phys0.5_kl0 의 epoch 4000 — 배포 구성(모델 -> 저역통과 -> 사영)으로
평가된 이력이 가장 길다. 모델을 바꾸려면 이 파일을 같은 구조의 다른 가중치로 바꿔 넣는다.
"""


def checkpoint_path():
    """`checkpoints/temp.pth` 의 절대 경로. 없으면 무엇을 어디에 두어야 하는지 알려 주고 실패한다."""
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"가중치 파일이 없습니다: {CHECKPOINT}\n"
            f"  checkpoints 폴더 안에 {CHECKPOINT_FILENAME} 을 두세요 (하위 폴더 없이).")
    return CHECKPOINT


def test_motion_files(limit=0):
    """held-out(test) 스플릿의 .pt 파일 절대 경로 목록.

    스플릿 규칙(SPLIT_SEED=42, VAL_RATIO=0.1)은 dataset_pipeline 의 것을 그대로 쓴다 —
    학습에 쓴 파일로 '라이브 시뮬레이션'을 하면 결과가 낙관적으로 오염된다.
    """
    from dataset_pipeline import get_split_files
    files = get_split_files(MOTIONS_DIR, split='test')
    files = [os.path.abspath(f) for f in files]
    return files[:limit] if limit else files
