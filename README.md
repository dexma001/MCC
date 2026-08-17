# MCC (Motion Capture Correction)

2026 Capsone Project

모션캡처(VMC) 결과에서 몸을 관통하는 **자기충돌(self-collision)을 제거**하되, 원래 동작의
의도(motion intent)는 보존하는 후처리 모델.
입출력 단위는 30프레임 윈도우의 87차원 텐서 = Hips 위치(3) + 21관절 로컬 쿼터니언(84).

> 코드에 남아 있는 `PVTVAE`는 프로젝트명이 아니라 **초기 VAE 아키텍처**를 가리키는 기술 용어다
> (`PVTVAE_baseline/`, 체크포인트 파일명 `pvtvae_epoch_*.pth` = 내부 구현 규약).

## 1. File Structure

### AI_model: main folder
- **dataset_pipeline**: 데이터 계약 + 학습 Dataset + 실행 경로 헬퍼 (전 모듈 공용 코어)
    + 뼈대 계약 상수: `PARENTS` / `BONE_NAMES` / `BONE_MAP` / `BONE_RADII`(캡슐 반지름)
    + `BandaiMotionDataset`, 결정론적 train/test 분할(`get_split_files`), 체크포인트 탐색 헬퍼
    + `RADII_MODE`(legacy / anatomical)로 캡슐 반지름 체계를 고르면 run 태그가 따라 붙는다
    + matplotlib / scipy / pandas를 로드하지 않는다 (무거운 의존성 금지)
- **preprocess**: Sample_Data(.csv) 파일을 .pt로 변환 — `python AI_model/preprocess.py`
- **corruption**: 손상(클리핑) 주입기. train(학습 입력)과 evaluate(평가 시나리오)가 공유한다
    + `transient`: sin-ramp 회전 (θ_peak 15~70°, 5~20프레임) — 센서 글리치 의미론
    + `persistent`: 상수 오프셋. 각도가 아니라 **관통 깊이 1~4cm**를 목표로 각도를 탐색한다
- **physics_module**: Define Loss Function (with Lumelsky's Algorithm)
    + FK(`forward_kinematics`)로 관절 월드 좌표 복원 + 캡슐-캡슐 거리 기반 충돌 손실
- **projection**: 미분가능 사영 레이어 — 네트워크 출력 뒤에서 관통을 기하학적으로 밀어낸다
    + 선형화 제약 사영(PBD 계열)을 반복해 `collision <= eps`를 강제한다
    + 손잡이: `PROJ_ENABLED` / `PROJ_K=8` / `PROJ_OMEGA=1.8` / `PROJ_MARGIN_CM=0.2` / `PROJ_PAIRS`
    + 위반 프레임이 없으면 즉시 종료 → 클린 입력에 대해 항등
- **models**: TransformerDenoiser (기본 아키텍처, 결정론적 — VAE 아님)
    + residual delta 예측 + 관절별 쿼터니언 재정규화, Hips 위치는 통과
    + `TransformerDenoiserCompat`: 평가·추론 진입점이 쓰는 아키텍처 중립 래퍼
- **train**: Training Function
    + 입력에만 손상을 주입하고 손실은 **클린 원본**과 계산한다 (`DECLIP_MODE`)
      → 혼합비 clean 50 / transient 15 / persistent 35
    + 손실 = `LAMBDA_RECON`·recon(MSE) + `lambda_phys`·collision, λ_phys는 에폭 커리큘럼
    + 실험 손잡이: `LAMBDA_PHYS`, `RUN_TAG_BASE`, `EPOCHS=100`, `BATCH_SIZE=32`, `lr=1e-4`
    + 감시 페어 `COLLIDING_PAIRS` 4개: 몸통-좌팔 / 몸통-우팔 / 좌팔-우팔 / 좌다리-우다리
    + 체크포인트는 run 설정별 폴더에 저장된다
- **inference**:
    + Input_0: 가지고 있는 Sample Data(maybe Clean)
    + Output: Model(Input_0) — 단발 산출용, CSV에 기록하지 않는다
- **demo_maker**:
    + Input_1: 가지고 있는 Random Sample Data(maybe Clean) + Artifical Collision
    + Output: Model(Input_1)
- **evaluate**: 시나리오별로 테스트셋을 평가해 `evaluate_results.csv`에 1행씩 기록
    + 시나리오: `clean`(항등 보존) / `legacy80`(LeftUpperArm +80°, 학습 분포 외 심층 손상) /
      `transient` / `persistent` — 평가 시드 고정으로 같은 손상이 재현된다
    + 평가할 실험은 train.py의 λ·run 태그 설정으로 고른다
    + Physical Plausibility: 충돌 손실, max/mean 관통 깊이(전체 및 감시 4페어), 깊이 제거율,
      무충돌 프레임 비율, bone length, jitter
    + Kinematic Accuracy: MPJPE, MAE(deg), 3DPCK 1/2/5/10cm(관절별·프레임별),
      cOKS(radii / uniform / COCO σ)
    + Motion Intent: `intent_dyn`(Δ² 동역학), `intent_mae_deg`
    + 성능: 윈도우 추론 시간 mean/p95, 프레임당 시간, device
    + 실행: `python evaluate.py [--corrupt | --scenarios clean,transient] [--limit N]`

### 루트
- **preprocess_motion.py**: 원본 .csv 정규화 (Unity 왼손 / Y-up / 쿼터니언).
  훼손 파일을 걸러내고, Hips 회전으로 up-axis를 판별해 보정한다
- **requirements.txt**: Python 3.11.9 / torch 2.2.2+cu121 등 고정 버전
- **evaluate_results.csv**: evaluate.py가 누적 기록하는 실험 로그
  (신규 컬럼은 맨 뒤에만 추가 — xlsx가 컬럼 위치를 참조한다)
- **evaluate_visualize.xlsx**: 위 CSV를 정리·시각화한 사본
- **checkpoints/log.txt**: 학습 로그
- **.gitignore** / **README**

### 로컬 전용 (원격 저장소에서 제외)
`.gitignore`에 등재되어 있다. clone만으로는 아래 도구가 실행되지 않는다.
- **AI_model/viz_motion.py**: 3D 모션 시각화 (`single` / `compare`)
    + 소스 상단 '소스 설정' 블록(`VIEW_MODE` 등)을 편집해 Run, 또는
      `python AI_model/viz_motion.py single [파일]` /
      `compare [--results DIR] [--save out.gif]` (인자가 소스 설정보다 우선)
    + 캡슐과 자기충돌 판정을 함께 표시한다
- **AI_model/viz_inject.py**: 손상 주입을 시각화 — Clean / Injected(침투 깊이=빨강 강도) /
  프레임별 깊이 그래프. 학습과 동일한 `corruption.py`를 호출한다
- **AI_model/viz_radii.py**: 캡슐 반지름(`BONE_RADII`) 점검·보정 도구
- **AI_model/3D_Collision_tot.py**: Collision 발생 시각화
- **AI_model/test_physics.py**: Testing physics_module
- **PVTVAE_baseline/**: 원본 PVTVAE(VAE) 재현용 위성 폴더. 데이터 계약·물리·손상 주입·평가
  스위트는 `AI_model/`을 import해 재사용하고 아키텍처(model.py)만 다르다
- 데이터·산출물: `Sample_Data/`, `*.pt`(processed_motions_VMC), `checkpoints/*.pth`, `*_results/`

## 2. Workflow
#### 현재 진행중인 모든 학습의 원본 데이터는 https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset 에 있음.

Sample_Data: Bandai_Dataset_csv_modi_tot 사용
- .csv 추출 과정: .bvh -> Blender (->.fbx) -> Unity(.csv)

실행 순서:
1. `python preprocess_motion.py` — 원본 .csv 정규화 (좌표계 / up-axis 보정)
2. `python AI_model/preprocess.py` — .csv → .pt
3. `python AI_model/train.py` — λ·run 태그 설정 후 학습
4. `python AI_model/evaluate.py` — 시나리오별 평가 → `evaluate_results.csv`
5. `python AI_model/inference.py` / `demo_maker.py` — 단발 결과 산출
