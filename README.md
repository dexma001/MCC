# PVTVAE(for Motion_Correction)

2026 Capsone Project

## 1. File Structure

#### AI_model: main folder  
- dataset_pipeline: Sample_Data(.csv) 파일을 .pt로 변환  
- physics_module: Define Loss Function (with Lumelsky's Algorithm)
- model: VAE model
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

## *Under this line, there is no classification folder*

watch_*_in_python: Sample_Data를 Matplot을 사용하여 확인  

Miscellaneous: Memo text
  
.gitignore
  
README
  

## 2. Workflow  
#### 현재 진행중인 모든 학습의 원본 데이터는 https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset 에 있음.

Sample_Data: Bandai_Dataset_csv_modi_tot 사용
- .csv 추출 과정: .bvh -> Blender (->.fbx) -> Unity(.csv)

## Done
+ 원본 데이터를 .csv로 가공 (Unity 좌표계에 맞는 왼손, Y up, Quarternion)
+ .csv -> .pt (for Machine Learning)
+ PVTVAE 구조 설계
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
    + Model: PVT+VAE -> TFM
        * TFM이 PVT+VAE에 비해 평균적인 성능이 더 좋았음
        * 현재 가장 Best Model: TFM 1/0.4
    + 학습 / 평가 시드 동일

    + 현재 평가 지표 개선 중 
        * 이 중, "Motion Intent Preservation" 부분에 대한 개선 필요.
    