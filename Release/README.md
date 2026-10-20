# Mocap_Retargeting_AI — 실시간 모션캡처 리타겟팅 AI 서비스

모션 캡처 스트리밍 중 생기는 **신체 자기 관통**(아래팔이 몸통을 뚫거나, 양 아래팔·양 정강이가 서로 파고드는 현상)을
실시간으로 리타겟팅 후 Warudo로 보내는 서비스입니다. 트래커와 Warudo 사이에서 동작하며, 양쪽 설정은 포트만 바꾸면 됩니다.

```
모캡 장비 ──VMC(UDP 39540)──▶ [ 보정 서비스 ] ──VMC(UDP 39539)──▶ Warudo
```

> 데모 영상: <!-- TODO: Warudo 3-아바타 비교(손상 입력 / 보정 출력 / 원본) 영상 링크 또는 GIF -->

# Index
- [요구사항과 호환](#요구사항과-호환)
- [빠른 시작](#빠른-시작)
  - [1. Download](#1-download)    
  - [2. Install](#2-install)    
  - [3. Self Test](#3-self-test-no-equipments-need)    
  - [4. Run](#4-run)    
  - [5. Control](#5-control)
- [폴더 구성](#폴더-구성)  

#### Engineering Documents
- [동작 방식](#동작-방식)
  - [관통 판정 기준](#관통-판정-기준)  
- [결과 한눈에](#결과-한눈에)
- [운영 안정성](#운영-안정성)
- [Bone Map](#bone-map)
- [알려진 한계](#알려진-한계)
- [학습·평가 및 데이터](#학습평가-및-데이터)
- [라이선스](#라이선스)


## 요구사항과 호환

| 항목 | 내용 |
|---|---|
| OS / 실행 환경 | Windows 10 이상, Python 3.11 (검증: 3.11.9, torch 2.2.2), **CPU 만으로 동작** |
| 프로토콜 | VMC (OSC over UDP). 입력 39540, 출력 39539 (Warudo 기본값) |
| 트래커 조건 | **전신 21개 본**(골반·척추·목·머리·어깨·팔·다리, [목록](#bone-map))을 모두 보내야 합니다. **하나라도 빠지면 Warudo 로 아무것도 나가지 않아 아바타가 움직이지 않습니다** (콘솔에 빠진 본 표시). 손가락·표정·눈은 그대로 통과 |
| 프레임률 | 30 fps 로 처리. 더 빠른 트래커는 최신 프레임만 사용 |
| 아바타 | 정규화 VRM (T-포즈에서 모든 본 회전 0) |
| 지연 | 출력 지연 67 ms (2프레임, 튐 완화용) + 처리 p95 약 15~25 ms. 시작·재연결 직후 1초는 원본 통과 |


## 빠른 시작

### 1. Download

```bash
git clone https://github.com/dexma001/Mocap_Retargeting_AI.git
cd Mocap_Retargeting_AI
```
이후 명령은 모두 이 폴더(저장소 최상위)에서 실행합니다.

### 2 Install

```bash
pip install -r requirements.txt
```

### 3. Self Test (No equipments need) 

```bash
python AI_Streaming/vmc_bridge.py --loopback-test
```

PC 안에서 가상 트래커 → 서비스 → 가상 Warudo 로 실제 UDP 왕복을 합니다. 보내는 동작은 **매 프레임 4쌍이 모두
2~5 cm 관통하는 합성 모션**이라, 설치·가중치·포트·좌표 변환과 함께 **관통이 실제로 지워지는지**까지 확인합니다.

```
[loopback] 관통 제거: 입력 4쌍 관통 2.0~5.2 cm (전 프레임) → 수신된 보정 프레임 61개의 4쌍 관통 max 0.000 cm
[loopback] OK: 88 프레임 왕복, 오류 0건 …
```

### 4. Run

```powershell
AI_Streaming\start_server.bat    # or AI_streaming\start_server.bat 더블클릭
```

- 트래커 앱의 VMC 송신 대상을 `127.0.0.1:39540` 으로 바꿉니다. 트래커가 Warudo 로 직접 보내면 스트림이 겹쳐 아바타가 떨립니다.
- Warudo: 캐릭터 → 모션 캡처 → VMC Receiver, 포트 39539.

**동작 확인** — 5초마다 콘솔(및 `AI_Streaming/logs/server.log`)에 출력됩니다:

```
in 1325 out 1323 | 처리 p95  14.3 ms max  24.4 | 예산초과 0.0% | … | 오류 0 | bypass=False | 트래커 1341f
    관통 4쌍 max 입력  8.62 → 출력  0.00 cm  [최근 5s]
```

`출력` 이 0.00 이면 Warudo 로 나간 프레임에 관통이 없다는 뜻입니다 (보낸 프레임을 다시 측정한 값).

> **고급: 브리지 직접 실행** — 설정을 시험하거나 문제를 볼 때는 supervisor 없이 브리지만 띄울 수 있습니다.  
> `python AI_Streaming/vmc_bridge.py --listen-port 39540 --send-port 39539`  
> 자동 재시작·로그 파일이 없고, 콘솔 키 `b`(보정 끄기/켜기)·`q`(종료)로 조작합니다.  
> `start_server.bat` 과 동시에 띄우면 같은 포트를 두고 충돌하므로 한쪽만 실행하십시오.  
> 옵션 전체는 `AI_Streaming/USAGE.md` §3.

### 5. Control

| 하고 싶은 것 | `start_server.bat` / `server.py` 로 실행 시 | `vmc_bridge.py` 를 직접 실행 시 |
|---|---|---|
| 보정 끄기/켜기 (원본 통과) | `python AI_Streaming/server.py --bypass on` / `off` | 콘솔에서 `b` |
| 상태 보기 | `python AI_Streaming/server.py --status` | 콘솔 통계 |
| 종료 | `python AI_Streaming/server.py --stop` 또는 창에서 Ctrl+C | 콘솔에서 `q` |

## 폴더 구성

```
Mocap_Retargeting_AI/
├── README.md                     
├── requirements.txt              pip install ist
│
├── AI_Streaming/                 ── 실시간 서비스 
│   ├── start_server.bat          실행 진입점 (server.py 를 UTF-8 콘솔로 띄움)
│   ├── server.py                 상시 실행 supervisor — bridge 감시·자동 재시작·로그
│   ├── server_config.json        server.py 설정 (포트, fps, 로그, 재시작 기준)
│   ├── control.py                상태·제어 HTTP (--status / --bypass / --stop 이 사용)
│   ├── vmc_bridge.py             서비스 본체 — VMC 수신 → 보정 → 송신, 자체 점검(--loopback-test)
│   ├── corrector.py              실시간 보정기 (링버퍼 → 모델 → 저역통과 → 사영)
│   ├── retarget.py               리그 ↔ VRM 좌표 변환
│   ├── rig_tpose.json            └ 변환에 쓰는 아바타 T-포즈 가정
│   ├── contract.py               프레임 규약 검사 (관절 순서·쿼터니언·단위)
│   ├── sinks.py                  VMC(OSC/UDP) 송신
│   ├── paths.py                  경로 기준점 
│   ├── USAGE.md                  옵션·상태 HTTP·문제 해결
│   ├── README.md                 동작 원리·실측 상세
│   └── logs/                     (실행 시 생성) server.log
│
├── AI_model/                     ── 보정 엔진 
│   ├── models.py                 ① Transformer Denoiser
│   ├── lowpass.py                ② 저역통과 필터
│   ├── projection.py             ③ 충돌 사영
│   ├── physics_module.py         FK 물리·관통 판정 (캡슐)
│   └── dataset_pipeline.py       데이터 규약·골격 정의·가중치 경로
│
├── checkpoints/
│   └── temp.pth                  모델 가중치
│
└── Sample_Data/
    └── Standard_BoneOffsets.csv  표준 골격 뼈 길이 (FK·관통 판정에 필수)
```

폴더 구조를 바꾸면 경로를 찾지 못합니다. 옵션·상태 HTTP·문제 해결은 `AI_Streaming/USAGE.md`,
동작 원리·실측 상세는 `AI_Streaming/README.md` 를 참고하십시오.

# Additional Documents (for Engineering)

## 동작 방식

```
트래커 프레임 ─▶ [30프레임 링버퍼] ─▶ ① 모델 ─▶ ② 저역통과 ─▶ ③ 충돌 사영 ─▶ Warudo
                  (첫 1초는 원본 통과)
```

| 단계 | 하는 일 | 무엇을 해결하나 |
|---|---|---|
| ① Transformer 디노이저 | 최근 1초(30프레임)를 보고 관절 21개의 회전량 보정 | **관통 제거**와 자연스러운 자세 복원  |
| ② 저역통과 필터 | 보정 결과의 Highpass frequency 제거 | Jittering 완화 |
| ③ 충돌 사영 (물리 기반) | 남은 관통을 최소 이동으로 밀어냄 (선형화 제약 사영, PBD 계열) | **안전망** — 필터가 다시 만든 관통과 모델이 드물게 남긴 관통을 0 으로 |

- **관통 제거의 주체는 ① 모델입니다.** ③ 사영은 실제로 전체 프레임의 0.05~0.47% 에만 개입합니다
  (손상 없는 입력에서는 0%).
- ③ 이 필요한 주된 이유는 ② 필터입니다. 떨림을 없애려고 프레임을 평활하면, 모델이 관통을 피해 둔 자세가
  이웃 프레임 쪽으로 끌려가 관통이 다시 생길 수 있습니다. 그래서 순서는 고정입니다 — 필터를 사영 뒤에 두면
  관통이 되살아납니다(실측 0.000 cm → 1.276 cm).
- 모델: 인코더·디코더 Transformer (d_model 128, 4 heads, 2+2 layers, 파라미터 2.55M, 가중치 10 MB).
  입력·출력 모두 `[30프레임, 87]` = 골반 위치 3 + 관절 21개 쿼터니언 84 (골반 위치는 그대로 통과).
- 학습: 공개 모션 데이터셋에 관통 손상(일시·지속)을 주입하고 원본으로 되돌리도록 학습.

### 관통 판정 기준

관통 판정 시 **표준 골격의 캡슐 근사 방식**을 사용합니다. 뼈마다 해부학적 신체 비율로 정한 반지름의
캡슐을 씌우고, 보장 대상 4쌍의 캡슐이 겹치는 깊이를 측정합니다.

| 보장 대상 (4쌍) | 대상 아님 |
|---|---|
| 왼 아래팔 ↔ 몸통 · 오른 아래팔 ↔ 몸통 · 아래팔 ↔ 아래팔 · 정강이 ↔ 정강이 | 손 ↔ 머리, 위팔 ↔ 몸통, 다리 ↔ 몸통, 의상·머리카락 등 메시 단위 관통 |

> 아바타 신장 차이 ±15% 까지는 판정에 영향이 없음을 실측했습니다. 극단적인 체형·큰 의상은 측정하지 않았습니다.

## 결과 한눈에

**오프라인 평가** — held-out 306개 구간(30프레임씩), 배포 가중치(`checkpoints/temp.pth`) 기준.
관통은 보장 대상 4쌍의 최대 깊이, MPJPE 는 손상 전 원본 대비 평균 관절 위치 오차입니다.

| 입력 시나리오 | 입력 관통 max | 모델만 통과 | **서비스 출력** | 원본 대비 MPJPE |
|---|---:|---:|---:|---:|
| 손상 없음 (clean) | 2.02 cm ¹ | — | **0.00 cm** | 0.49 cm |
| 일시적 관통 (transient) | 8.83 cm | — | **0.00 cm** | 0.84 cm |
| 지속적 관통 (persistent) | 8.92 cm | 2.68 cm ² | **0.00 cm** | 0.78 cm |
| 학습 분포 밖 팔 충돌 (legacy80) | 8.87 cm | 3.73 cm ² | **0.00 cm** | 1.60 cm |

¹ 손상을 넣지 않은 원본 모션 데이터 자체에도 관통이 있습니다.
² 모델만으로 관통 깊이의 99.5~99.6% 를 제거합니다(persistent 기준 관통 없는 프레임 23.7% → 99.5%).
  표의 값은 남은 0.5% 프레임 중 가장 깊은 것입니다.

**실시간 경로 실측** — 별도 프로세스의 가상 트래커 → UDP → 서비스 → 수신기 (persistent 손상, 55초, CPU):

| 항목 | 값 |
|---|---|
| 입력 관통 max → 출력 관통 max | 8.62 cm → **0.00 cm** (보정 프레임 전부) |
| 프레임 처리 시간 p95 | 약 14 ms (30fps 예산 33.3 ms, 초과 0%) |
| 보정 예외 / 송신 실패 | 0 / 0 |

장비 없이 같은 확인을 10초 만에 직접 해 볼 수 있습니다 → [자체 점검](#3-self-test-no-equipments-need).

## 운영 안정성

방송 중 서비스가 문제를 일으켜도 **방송은 끊기지 않게** 설계했습니다.

- 보정 중 예외가 나면 그 프레임은 원본을 그대로 보냅니다.
- 브리지가 죽거나 멈추면(heartbeat 15초 무응답) supervisor 가 자동으로 다시 띄웁니다.
- 처리 시간이 예산(33 ms)에 가까워지면 사영 반복 횟수를 줄여 프레임을 놓치지 않습니다.
- 트래커가 끊기면 송신을 멈추고(Warudo 는 마지막 포즈 유지), 다시 연결되면 자동으로 이어집니다.
- 중복 실행 방지, 크기 제한 회전 로그, 로컬 전용 상태·제어 HTTP.
- 장애 주입 시험 10/10 통과 (기동, 중복 실행 거부, 원격 bypass, 강제 종료 후 재시작, 트래커 재연결 등).

## Bone Map

모델이 다루는 본은 아래 21개입니다. 이름은 Unity HumanBodyBones / VMC 철자 그대로이고, `[n]` 은 모델 텐서 안의 관절 번호입니다.

```
Hips [2]  (루트 — 위치도 함께 전달)
├── Spine [20]
│   └── Chest [0]
│       ├── Neck [11]
│       │   └── Head [1]
│       ├── LeftShoulder [7]
│       │   └── LeftUpperArm [9]
│       │       └── LeftLowerArm [5]
│       │           └── LeftHand [4]
│       └── RightShoulder [16]
│           └── RightUpperArm [18]
│               └── RightLowerArm [14]
│                   └── RightHand [13]
├── LeftUpperLeg [10]
│   └── LeftLowerLeg [6]
│       └── LeftFoot [3]
│           └── LeftToes [8]
└── RightUpperLeg [19]
    └── RightLowerLeg [15]
        └── RightFoot [12]
            └── RightToes [17]
```

**텐서 배치** — 한 프레임은 87차원입니다.

| 범위 | 내용 |
|---|---|
| `[0:3]` | Hips 위치 (m) — 모델이 바꾸지 않고 그대로 통과 |
| `[3 + 4n : 7 + 4n]` | 관절 n 의 로컬 회전 쿼터니언 `(x, y, z, w)`. n 은 본 이름의 알파벳순 (Chest = 0 … Spine = 20) |

**관통 보장 4쌍이 쓰는 구간** (캡슐 = 두 본을 잇는 선분)

| 쌍 | 구간 A | 구간 B |
|---|---|---|
| 왼 아래팔 ↔ 몸통 | LeftLowerArm → LeftHand | Hips → Chest |
| 오른 아래팔 ↔ 몸통 | RightLowerArm → RightHand | Hips → Chest |
| 아래팔 ↔ 아래팔 | LeftLowerArm → LeftHand | RightLowerArm → RightHand |
| 정강이 ↔ 정강이 | LeftLowerLeg → LeftFoot | RightLowerLeg → RightFoot |

**21개 밖의 본**

- `UpperChest` — 트래커가 보내면 Chest 에 합쳐(Chest·UpperChest) 모델에 넣고, 보낼 때 원래대로 나눕니다.
- 손가락·눈·턱, 표정(BlendShape), Root — 보정하지 않고 그대로 통과합니다.
- 21개 중 하나라도 들어오지 않으면 프레임을 만들지 않고 Warudo 로 아무것도 보내지 않습니다. 콘솔에 `수신 대기… (빠진 본 [...])` 가 5초마다 찍힙니다.

## 알려진 한계

| 한계 | 현황 |
|---|---|
| 멀쩡한 동작도 약간 바뀜 | 실시간 경로에서 손상 없는 입력도 평균 약 1.6 cm 움직입니다 (모델 재학습 과제). 위 오프라인 표의 0.49 cm 보다 큰 것은 실시간에서는 미래 프레임이 거의 없는 구간 끝쪽 출력을 쓰기 때문입니다 (1.6 cm 는 출력 지연 적용 전 측정) |
| 간헐적 튐 | 1000프레임당 6 cm 이상 튐: 원본 4회, 보정 출력 25회. 출력 지연으로 절반으로 줄였지만 남아 있습니다 |
| 보장 범위 | 위 4쌍만. 그 밖의 전신 관통은 4~6 cm 남을 수 있습니다 |
| 지연 상한 | 평균·p95 는 예산 안이지만 최악의 경우 상한은 보장하지 않습니다 (사영 비용이 입력에 따라 다름) |
| 본 롤 | 손바닥 방향·팔 롤·발 기울기는 아바타의 T-포즈 가정(`rig_tpose.json`)에 따라 어긋날 수 있습니다 |
| 미검증 | 60 fps 이상 트래커, 프리즈·텔레포트·트래커 소실 같은 손상 유형 |


## 학습·평가 및 데이터

- 학습·평가·시각화 코드와 전체 실험 기록: https://github.com/dexma001/MCC
- 학습 데이터: <!-- TODO: 데이터셋 정식 명칭·출처 링크 --> (이 저장소에는 포함하지 않습니다)

## 라이선스

<!-- TODO: 코드 라이선스, 그리고 학습 데이터 이용 조건에 따른 가중치 배포 조건 -->
