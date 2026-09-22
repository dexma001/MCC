# AI_model_large — 용량 축 실험 환경

`AI_model`과 **같은 손실·같은 데이터·같은 주입**으로 학습하되 **아키텍처만 키운** 대조 환경이다.
`claude_analysis/projection_necessity_analysis_20260911.md` §3이 남긴 공백
("용량 스윕을 한 번도 하지 않았다")을 1런으로 메우는 것이 목적이다.

## 파일

| 파일 | 역할 | AI_model의 대응 |
|---|---|---|
| `models_large.py` | 확장 아키텍처 + 크기 상수 + `arch_tag()` | `models.py` |
| `train_large.py` | 학습 루프 (손실·커리큘럼·저장 규칙 동일) | `train.py` |
| `evaluate_large.py` | **얇은 드라이버** — `evaluate.py`를 복사하지 않고 전역만 바꿔 호출 | `evaluate.py` |

`dataset_pipeline` / `physics_module` / `corruption` / `lowpass` / `projection`은 **복사하지 않고
`AI_model`의 원본을 import**한다. 복사본은 조용히 갈라지고, 갈라지는 순간 두 실험은 비교 불가능해진다.
**`AI_model`의 어떤 파일도 수정하지 않는다.**

### ⚠️ 파일명이 `models.py` / `train.py`가 아닌 이유

파리티를 맞추려다 되돌렸다. 이 폴더의 스크립트를 실행하면 `sys.path[0]`이 이 폴더가 되어
**`AI_model`의 동명 모듈을 가린다.** 그러면 `AI_model/evaluate.py` 안의
`from models import TransformerDenoiserCompat`가 이 폴더의 `models.py`를 잡아 ImportError로 죽고,
`from train import ...`는 조용히 이 폴더의 상수를 읽는다. 이름을 분리하면 **sys.path 순서가
결과를 바꾸지 않는다.**

## 기본 구성

| | 기준선 (`AI_model`) | 확장 (여기) |
|---|---|---|
| d_model | 128 | **256** |
| nhead | 4 | **8** |
| num_layers (enc=dec) | 2 | **4** |
| dim_feedforward | 2048 (명시 안 된 PyTorch 기본값, 16배) | **1024 (명시, 4배)** |
| latent_dim | 64 | 64 (**동일 유지**) |
| 파라미터 | 2,547,220 | **7,457,684** (2.93배) |

`latent_dim`은 일부러 고정했다 — 실효 랭크가 14~18이라 병목이 아니고, 늘리면 "용량"이라는
단일 변수가 두 개가 된다.

**손실 설정은 기준선과 동일**: `LAMBDA_PHYS=0.3`, MSE 배수 1.3, 손상 시드 777, anat 반지름.
⇒ 짝이 되는 소형 대조군: `checkpoints/tfm_declip_cov_MSEOnly_1.3_anat_recon1_phys0.3_kl0/`
(이미 학습돼 있다). 이 짝이 성립해야 "아키텍처만 다른 A/B"가 된다.

## 실행

```bash
# 1) 크기·파라미터 확인 (학습 없음)
python AI_model_large/models_large.py

# 2) 학습 (100 epoch, 10 epoch마다 체크포인트)
python AI_model_large/train_large.py

# 3) 평가 — 배포 구성(저역통과 + 사영)으로 evaluate_results.csv에 행 추가
python AI_model_large/evaluate_large.py

# 스모크(기록 없이 파일 3개만)
python AI_model_large/evaluate_large.py --limit 3 --no-csv
# 모델 단독 성능(사영·필터 끄기)
python AI_model_large/evaluate_large.py --no-lowpass --no-projection
```

## 크기를 바꿀 때

`models_large.py` 상단의 `LARGE_*` 상수만 고치면 된다. `arch_tag()`가 자동으로 바뀌어
**run 폴더가 갈린다** (`..._d256h8L4f1024_anat_recon1_phys0.3_kl0`). 이것이 이 환경의 핵심
안전장치다 — 구조가 다른 `.pth`가 한 폴더에 섞이면 복구가 불가능하다.
추가로 `train_large.py`가 폴더의 `run_config.json`을 읽어 아키텍처가 다르면 **즉시 중단**한다.

후보 구성별 파라미터 수는 `python AI_model_large/models_large.py`가 표로 출력한다.

## 읽을 때 주의

- **평가 CSV의 행은 소형 런과 같은 파일에 쌓인다.** `run_tag`에 `tfmL_`와 아키텍처 태그가
  들어가므로 구분은 되지만, 비교할 때는 **같은 λ·같은 반지름 시대** 행끼리만 봐야 한다.
- 첫 판정 기준은 관통이 아니라 **충실도(MPJPE/cOKS)와 clean do-no-harm**이다. 4쌍 관통 제거는
  사영이 모델과 거의 무관하게 100%를 만들기 때문에 모델을 가르지 못한다(30런 실측).
- 용량이 효과를 낸다면 나타날 곳: `proj_frames_pct` 감소, clean `max_pen_after` 감소,
  `collateral_pos_cm` 감소. 이 세 열을 먼저 보라.
