# MCC (Motion Capture Correction)

2026 Capsone Project

> **프로젝트명 이력**: 이 프로젝트는 **PVTVAE → MCC**로 개명되었다.
> 다만 코드·문서에 남아 있는 `PVTVAE`는 대부분 **프로젝트명이 아니라 초기 VAE 아키텍처**를
> 가리키는 기술 용어이므로 그대로 둔다 (예: "PVTVAE 대비 바뀐 것", `PVTVAE_baseline/`).
> 마찬가지로 체크포인트 파일명 `pvtvae_epoch_*.pth`는 **내부 구현 규약**이라 개명하지 않는다
> — 디스크의 기존 가중치 100개 및 `dataset_pipeline.py`의 탐색 헬퍼와 맞물려 있다.

## 1. File Structure

#### AI_model: main folder  
- dataset_pipeline: 데이터 계약 상수 + 학습 Dataset + train/test 분할 + 체크포인트 헬퍼
    + 전 모듈이 import하는 코어. matplotlib/scipy/pandas를 로드하지 않는다.
- preprocess: Sample_Data(.csv) 파일을 .pt로 변환 (`python AI_model/preprocess.py`)
- viz_motion: 3D 시각화 — 실행 방법 2가지
    + **소스 설정 + VS Code 실행**: 파일 상단 '소스 설정' 블록의 `VIEW_MODE`
      (`"single"`/`"compare"`)와 관련 값을 편집한 뒤 그냥 Run(Ctrl+F5).
      인자를 주지 않으면 이 설정이 쓰인다. F5로는 `.vscode/launch.json`의
      모드별 구성(COMPARE / SINGLE / gif 저장 / inference_results)을 골라 실행.
    + **명령줄 인자**(소스 설정보다 우선, 일회성):
      `python AI_model/viz_motion.py single [파일]` /
      `python AI_model/viz_motion.py compare [--results DIR] [--save out.gif]`
- viz_inject: 손상 주입(클리핑)이 어떻게 들어가는지 gif로 시각화
    + `python AI_model/viz_inject.py both` — transient/persistent gif 모두 생성
      (`demo_results/injection_*.gif`). `--seed` 고정 시 같은 gif가 재현된다.
    + 학습과 동일한 `corruption.py` 주입기를 호출하므로 실제 학습 입력 그대로다.
    + 화면: Clean(주입 대상 본 초록) / Injected(침투 깊이=빨강 강도) / 프레임별 깊이 그래프
      → transient=봉우리 하나, persistent=구간 내내 고원. 유형 차이가 그래프로 드러난다.
- physics_module: Define Loss Function (with Lumelsky's Algorithm)
- models: TransformerDenoiser (기본 아키텍처, 결정론적 — VAE 아님)
    + 원본 PVTVAE(VAE 포함)는 PVTVAE_baseline/ 으로 분리 보존됨
- train: Training Function
- inference:  
    + Input_0: 가지고 있는 Sample Data(maybe Clean)
    + Output: Model(Input_0)
- demo_maker: 
    + Input_1: 가지고 있는 Random Sample Data(maybe Clean) + Artifical Collision
    + Output: Model(Input_1)
- evaluate: 평가지표:
    + Physical Plausibility
        * Collision Depth: 보정된 결과의 Collision (loss)
        * Bone Length Jitter: 보정 과정의 관절 위치 변화가 미친 영향
    +  Kinematic Accuracy
        * MPJPE: Motion 보존 평균 오차
  
- 3D_Collision_tot: Collision 발생 시각화
- test_physics: Testing physics_module

#### PVTVAE_baseline: 원본 PVTVAE(VAE) 재현용 위성 폴더
- 데이터 계약/물리/손상 주입/평가 스위트는 AI_model/ 을 그대로 import해 재사용하고,
  아키텍처(model.py)만 다르다. run 폴더 태그로 구분되어 결과가 서로 덮어쓰지 않는다.

## *Under this line, there is no classification folder*

.gitignore
  
README
  

## 2. Workflow  
#### 현재 진행중인 모든 학습의 원본 데이터는 https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset 에 있음.

Sample_Data: Bandai_Dataset_csv_modi_tot 사용
- .csv 추출 과정: .bvh -> Blender (->.fbx) -> Unity(.csv)

## Done
+ 원본 데이터를 .csv로 가공 (Unity 좌표계에 맞는 왼손, Y up, Quarternion)
+ .csv -> .pt (for Machine Learning)
+ PVTVAE 구조 설계 (초기 아키텍처 — 현재 기본은 TransformerDenoiser)
+ 100 Epochs 학습 (처음 20 Epochs는 KL만, 이후 PL(0~0.1 for 20 epochs))  
  
- 2026.07.06
    + 원본 데이터 정규화 (preprocess_motion.py / processed_motions_VMC)
    + AI_model/* 수정
    + jitter_loss 삭제 ()
    + epoch 100 / 200으로 lambda_phys 값 수정하여 결과 확인

- 2026.07.25
    + 학습 과정에 의도적 Collision 추가
        * clean: collision이 없는 깨끗한 Data
        * transient: 팔이나 다리가 의도적으로 15~70도 안으로 굽게 됨(센서 오류 등을 의도)
        * persistent: Capsule 거리가 1~4cm인 지속적 collision
            + clean : collision = 5 : 5
            + collision -> transient : persistent = 3: 7
    + 평가 과정에 clean / transient / persistent + legacy80
        * legacy80: leftupperarm이 80도 안으로 (학습 과정에 없었던 collision)
    + Model: PVT+VAE -> TFM (PVE+VAE Model은 local에 저장)
        * TFM이 PVT+VAE에 비해 평균적인 성능이 더 좋았음
        * 현재 가장 Best Model: TFM 1/0.3
    + 학습 / 평가 시드 동일

    + 현재 평가 지표 개선 중 
        * 이 중, "Motion Intent Preservation" 부분에 대한 개선 필요.
    