# AI_Streaming — 실시간 보정 런타임 + 라이브 시뮬레이터 + Warudo(VMC) 브리지

`checkpoints/`의 학습된 모델을 **한 프레임씩 들어오고 나가는 형태**로 감싼 배포 런타임과,
실제 송신을 붙이기 전에 그 환경을 재현해 판정하는 시뮬레이터, 그리고 **Steam Warudo** 로
실제 송수신하는 VMC 브리지다. **`AI_model`의 어떤 파일도 수정하지 않는다** — 원본을 import 해서 쓴다.

## 파일

| 파일 | 역할 |
|---|---|
| `paths.py` | 경로 부트스트랩 (AI_model을 sys.path에 얹고, 체크포인트/데이터 절대경로 제공) |
| `contract.py` | 87차원 입출력 계약 — 관절 순서·쿼터니언 규약·단위·검증·학습 코드와의 일치 검사·Unity 본 이름 검사 |
| `corrector.py` | **런타임 본체** `StreamingCorrector.push(frame) -> (frame, info)` |
| `retarget.py` | **리그 로컬 ↔ VRM 정규화 로컬 변환** (Warudo 호환의 핵심, 아래 참조) |
| `derive_rest_pose.py` | 리그 T-포즈(`rig_tpose.json`) 도출 + 7항목 검증. 데이터가 바뀌면 다시 실행 |
| `rig_tpose.json` | 도출된 리그 T-포즈 전역 회전 21개 (retarget 이 읽는다) |
| `sources.py` | 입력원 — held-out 모션을 실시간처럼 재생 (clean/transient/persistent/legacy80) |
| `sinks.py` | 출력부 — Null / Record / **VmcOscSink** (VMC 번들 생성, 기본 DRY-RUN) |
| `vmc_bridge.py` | **실제 송수신**: 트래커(VMC) → 보정 → Warudo(VMC). 루프백 자체검사·데이터셋 재생·프레임 로그 포함 |
| `tracker_sim.py` | **가상 모션캡처 장비**: held-out 데이터셋(+손상 주입)을 별도 프로세스에서 VMC/UDP 로 실시간 송신 |
| `selftest.py` | 출시 전 자체 점검 16항목 (T1~T10 런타임, T11~T15 Warudo 호환, T16 출력 시간축) |
| `run_sim.py` | 라이브 시뮬레이션 + 지연·관통·do-no-harm 판정 보고 |
| `server.py` | **상시 실행 supervisor** — 브리지를 자식으로 띄워 크래시·멈춤 시 재시작, 회전 로그, 중복 실행 방지, 원격 상태·제어 |
| `server_config.json` | 서버 설정 (체크포인트 에폭 고정, 포트, torch 스레드 수, 감시 주기) |
| `control.py` | 브리지의 로컬 상태·제어 HTTP (`/health` `/status` `POST /bypass /reset /shutdown`) + 클라이언트 |
| `start_server.bat` | 서버 실행 배치 (더블클릭 / 작업 스케줄러용) |
| `USAGE.md` | **파일별 사용법 상세** — CLI 옵션 전체, Python API, 작업 순서, 출력 읽는 법, 문제 해결 |

```bash
python AI_Streaming/selftest.py                                   # 먼저 이것 (16항목, 소켓 루프백 포함)
python AI_Streaming/run_sim.py --scenario clean   --frames 600
python AI_Streaming/run_sim.py --scenario persistent --frames 600
python AI_Streaming/run_sim.py --scenario persistent --no-adaptive-k   # 예산 가드 끄고 대조
python AI_Streaming/run_sim.py --pace --osc-dry-run                # 실시계 + VMC 패킷 생성
python AI_Streaming/run_sim.py --large --run <확장런폴더>          # AI_model_large 모델로

python AI_Streaming/vmc_bridge.py --loopback-test                  # Warudo·데이터셋 없이 UDP 왕복 + 관통 제거 검사
python AI_Streaming/vmc_bridge.py --replay clean                   # 데이터셋 모션을 Warudo(39539)로 재생 (프로세스 내)
python AI_Streaming/vmc_bridge.py --listen-port 39540              # 라이브: 트래커 → 보정 → Warudo
python AI_Streaming/tracker_sim.py --scenario persistent           # (다른 창) 가상 트래커 → 브리지(39540)

python AI_Streaming/server.py                                      # 상시 실행 (로컬 서버, USAGE.md §9)
python AI_Streaming/server.py --status | --bypass on | --stop      # 다른 창에서 조회·조작
```

## 데이터셋으로 Warudo 실측하기 — "장비를 착용하고 보내는 것처럼"

`tracker_sim.py` 가 **별도 프로세스에서 자기 시계로 UDP 송신**하므로 브리지·Warudo 는 실제 장비와
구분하지 못한다. 같은 프레임을 여러 포트로 보내 Warudo 에서 캐릭터를 나란히 비교한다.

**Warudo 준비** (VMC Receiver 는 여러 개 둘 수 있고 포트만 달라야 한다):

| Warudo 캐릭터 | VMC Receiver 포트 | 받는 것 |
|---|---|---|
| A (손상 입력) | 39541 | tracker_sim 이 보내는 손상 주입 프레임 (브리지와 같은 입력) |
| B (보정 출력) | 39539 | 브리지 출력 = 모델 → 저역통과 → 사영 |
| C (클린 참조) | 39542 | 손상 주입 전 원본 (정답) |

캐릭터 → 모션 캡처 → VMC Receiver 를 각 캐릭터에 추가하고 포트를 위처럼 맞춘다. 캐릭터 하나만
쓰려면 B(39539)만 만들면 된다. 모델은 VRM(정규화 본).

**실행** (창 2개):

```bash
# 창 1 — 브리지 (모델 로드·워밍업 후 "수신 대기…" 가 뜨면 준비 완료)
python AI_Streaming/vmc_bridge.py --listen-port 39540 --send-port 39539 --log-csv bridge_log.csv

# 창 2 — 가상 트래커
python AI_Streaming/tracker_sim.py --scenario persistent --loop \
    --to 127.0.0.1:39540 --to 127.0.0.1:39541 --clean-to 127.0.0.1:39542 --log-csv tracker_log.csv
```

시나리오: `clean`(do-no-harm — B 가 A 와 얼마나 다른가), `persistent`/`transient`(주입 관통 제거),
`legacy80`(80° 팔 주입, 학습 분포 밖). `--fps 60` 으로 60fps 장비를 흉내 내면 브리지의
sample-and-hold 리샘플링을 검사할 수 있다. `--hips-height` 로 아바타 골반 높이를 맞춘다
(데이터셋 Hips 는 첫 프레임 상대값이라 그대로면 바닥에 파묻힌다).

**무엇이 실측되는가**

* 브리지 콘솔(5초마다): 처리 p95/max ms, 예산 초과 %, K 깎임, hold(같은 프레임 재사용), 오류, bypass,
  **관통 4쌍 max 입력 → 출력**(받은 프레임 vs 보낸 프레임을 FK 로 재측정, 최근 5초).
  tracker_sim 콘솔의 관통은 **보정 전 입력**이라 줄지 않는다(최근 5초 / 누적 둘 다 표시).
* `--log-csv`(브리지): 프레임별 `t_s, emitted_stage(warmup/corrected/bypass), ms, k_used, k_capped,
  proj_residual_cm, hold, error, pen_in_cm, pen_out_cm`. `--log-csv`(tracker_sim): 프레임별 입력 4쌍 관통 cm 와 클린 관통 cm.
  두 로그를 `t_s` 로 맞추면 "이 관통이 들어왔을 때 보정에 몇 ms 걸렸고 잔여가 얼마였나"가 나온다.
* Warudo 화면: A(손상)·B(보정)·C(정답)를 나란히 — 관통 제거, 팔 롤/손바닥/발 기울기(T-포즈 자유도),
  Hips 높이, 지연(A 와 B 의 시간차 ≈ 워밍업 1초 + 처리 시간).

2026-09-16 종단 검사(대역 수신기 3개, Warudo 대신, 당시 기본 로드 epoch 500): 가상 트래커 29.4fps 송신 → 세 포트 모두 수신,
persistent 입력 관통 max 8.5cm → 보정 출력 0.00cm, Hips 절대 위치 복원, 브리지 처리 p95 22.8ms.

**동작 규칙**: 트래커 수신이 0.5초 없으면 브리지는 송신을 멈춘다(Warudo 는 마지막 포즈 유지 —
held 프레임을 계속 보내지 않는다). 2초 이상 끊겼다 돌아오면 세션(정규화 기준·링버퍼)을 리셋하고
첫 1초는 원본 통과. `--no-retarget` 은 변환 없이 보내 **뒤틀림을 시연**하는 용도다.

## 파이프라인

```
입력 프레임 --> [30프레임 링버퍼] --> 모델 --> 저역통과 --> 사영 --> 출력 프레임
                    |
                    +-- 버퍼가 찰 때까지(29프레임) 원본 그대로 통과
```

순서는 취향이 아니라 강제다. 필터를 사영 뒤에 두면 관통이 되살아난다
(selftest T8이 매 실행 실측으로 재확인한다: 필터→사영 0.000cm vs 사영→필터 1.276cm).

라이브(Warudo)에서는 앞뒤에 변환이 붙는다:

```
트래커(VMC) → [VRM→리그 변환 + 세션 정규화] → 위 파이프라인 → [정규화 되돌림 + 리그→VRM 변환] → Warudo(VMC)
                                                손가락·표정·눈·UpperChest 는 그대로 통과 ─────────────▶
```

## Warudo 호환 — 검사 결과와 결정 사항

### 1. 쿼터니언 규약 불일치 (가장 중요, 해결함)

Warudo 의 VMC 수신은 **"T-포즈에서 모든 본의 로컬 회전이 (0,0,0)" 인 정규화 모델(VRM)** 을
전제한다. 이 프로젝트의 87차원 텐서에 든 쿼터니언은 그 규약이 **아니다** — 원본 CSV 가 FBX
리그의 로컬 회전이라 관절 방향이 내장돼 있다 (정지 자세에서 Spine ≈ Y축 90°, UpperLeg ≈ Z축
180°, 리그 T-포즈 로컬 최대 180°). 그대로 보내면 아바타가 완전히 뒤틀리고, 트래커의 VRM 규약
값을 그대로 모델에 넣으면 학습 분포 밖의 입력이 된다.

`retarget.py` 가 양방향 변환을 한다: `V_b = G0_parent · L_b · inv(G0_b)`. 이 변환은
**모든 관절의 월드 위치를 정확히 보존**하고(selftest T13: 최대 오차 0.00008cm), 각 본의 월드
회전이 독립이라 오차가 사슬을 따라 누적되지 않는다.

`G0`(리그의 T-포즈)는 어디에도 정의돼 있지 않아 `derive_rest_pose.py` 가 **기하로 구성**했다:
척추 +Y·팔 ±X·다리 −Y 방향 조건 + Hips/Chest 는 좌우 자식 오프셋으로 비틀림 고정 + 나머지는
데이터셋(27,765 서 있는 프레임)의 평균 자세에서 최소 회전. 검증: 팔 수평 편차 0.00°, 위치 보존,
왕복 일치, 좌우 대칭(중앙값 차 ≤ 10.5°), 서 있는 자세에서 Head VRM 회전 중앙값 10°, Spine 5°.

**남는 자유도**: 뼈 축 둘레의 비틀림(롤)은 기하가 결정하지 못한다. 위치는 맞지만 손바닥
방향·팔 롤·발 기울기(리그 43°)가 아바타와 다를 수 있다 → Warudo 에서 `--replay clean` 으로
**눈으로 확인**해야 한다. 틀리면 `derive_rest_pose.py` 의 `TPOSE_DIR`/비틀림 규칙을 조정한다.

### 2. "Normalized" 의 정의 (배포 갭 보고서 P2-2, 확정)

`Bandai_Dataset_csv_VMC` 와 `_Normalized` 를 대조해 실측: 파일마다 **상수 월드 회전 R**
(첫 프레임 앞 방향을 정면으로 맞추는 yaw, 일부 파일은 Z-up/뒤집힘 보정 포함)을 Hips 쿼터니언에
곱하고, Hips 위치를 `R·(p − p0)` 로 두었다(오차 1e-15). 다른 20개 본은 불변. 즉 학습 데이터의
Hips 위치는 **첫 프레임 기준 상대값(≈0)** 이다.

`vmc_bridge.SessionNormalizer` 가 라이브에서 같은 규약을 재현한다(첫 프레임 Hips 원점화 + 다리
위치로 잰 앞 방향을 +Z 로). 모델 입력에만 적용하고 출력에서 되돌리므로 Warudo 는 트래커의
절대 위치를 그대로 받는다(루프백 T15 가 확인).

### 3. VMC 패킷 (Warudo 기본 포트 39539)

프레임당 OSC 번들 1개: `/VMC/Ext/OK 1 3 0 1` → `/VMC/Ext/Root/Pos` → `/VMC/Ext/Bone/Pos` × 21
(**Hips 만 위치**, 나머지 위치 0, 단위 쿼터니언) → pass-through → `/VMC/Ext/T`. 본 이름 21개는
전부 Unity HumanBodyBones 철자(T11). UpperChest 는 리그에 없다 — 트래커가 보내면 Chest 에
접어 모델에 넣고, 출력에서 다시 풀어 UpperChest 는 원본을 통과시킨다.

### 4. 배선 (Warudo 쪽)

* Warudo: 캐릭터 → 모션 캡처 → **VMC** 수신, 포트 39539, **VRM 모델**(정규화 본) 사용.
  Warudo 문서: 모델은 T-포즈에서 본 회전 0 이어야 하며, 옛 VRM 은 Enforce T-Pose 로 재출력.
* 트래커(VMC 앱·VSeeFace·mocopi 등)의 송신 대상을 Warudo 가 아니라 **브리지의 `--listen-port`** 로.
  트래커가 Warudo 로 직접 보내면서 브리지도 보내면 두 스트림이 겹쳐 아바타가 떤다.
* Warudo 의 **VMC Sender** 에셋을 입력으로 쓸 수 있으나, 같은 캐릭터가 VMC 수신도 하면
  보정값이 다시 입력되는 **되먹임 고리**가 된다. 입력·출력 캐릭터를 분리하거나 외부 트래커를 쓴다.
* 확인 순서: `selftest.py` → `vmc_bridge.py --loopback-test` → `--replay clean` 을 Warudo 에서
  눈으로(팔 롤·손바닥·발 기울기·Hips 높이 `--replay-hips-height`) → 라이브.

### 5. 유의사항 (런타임이 해결하지 않는 것)

* 트래커 fps ≠ 30 이면 sample-and-hold 로만 맞춘다(보간 없음). 60fps 트래커는 절반을 버린다.
* 트래커가 2초 이상 끊기면 세션(정규화 기준·링버퍼)을 리셋한다 → 재개 첫 1초는 원본 통과.
* 보정 예외는 원본 통과로 흡수한다(방송이 죽지 않는다). `b` 키로 bypass, `q` 로 종료.
* Hips 절대 높이는 트래커 값을 그대로 쓴다. 데이터셋 재생은 `--replay-hips-height`(기본 0.9m).
* 아바타 신장 차이는 ±15% 까지 관통 판정에 무해(2026-09-04 실측). 극단 체형은 미측정.
* **체크포인트 에폭**: `paths.checkpoint_path()` 는 폴더의 **가장 큰 에폭**을 고른다. 기본 런
  `tfm_declip_cov_l10.1_anat_recon1_phys0.5_kl0` 폴더에 4000 에폭까지 쌓여 있어 지금은
  `pvtvae_epoch_4000.pth` 가 로드된다(아래 실측 표 기준). 폴더에 더 큰 에폭이 쌓이면 **아무 경고 없이
  로드 대상이 바뀐다** — 재현이 필요하면 `--epoch 4000` 처럼 고정한다. 예전 표(epoch 100)를 재현하려면 `--epoch 100`.
* `stride>1` / `emit_lag>0` 의 출력 지연은 `stride-1+emit_lag` 프레임이다(스트림 첫 그만큼은 출력 없음).
  라이브(브리지)는 stride=1, **emit_lag=2** = 추가 지연 2프레임(67ms)으로 돈다(2026-10-07, `--emit-lag 0` 이면 예전처럼 0).

## 배포 갭 보고서에서 여기서 해결한 것

| 항목 | 해결 방식 |
|---|---|
| **P0-1** 배선 없음 | `corrector.py` + `sources/sinks` + **`vmc_bridge.py`** 로 전 구간 배선 |
| **P0-2** `inference_mode` 크래시 | 생성 시 검사해 즉시 중단 (selftest T4) |
| **P0-3** 30프레임 미만 | 워밍업 바이패스 (selftest T5, 비트 동일 통과) |
| **P1-2** 지연 상한 없음 | 남은 예산으로 사영 반복 K를 깎는 `adaptive_k` |
| **P2-1** 계약 검증 없음 | `contract.validate_frame` + 주기 검사 + 학습 코드 대조 |
| **P2-2** "Normalized" 정의 없음 | 원본/정규화 CSV 대조로 **확정** + `SessionNormalizer` 로 재현 |
| **P3-1** 패닉 스위치 없음 | `corrector.bypass = True` / 브리지 `b` 키 |
| 콜드 스타트 | 생성자에서 1회 워밍업 — 첫 위반 프레임 스파이크 30→13ms |
| (신규) VRM 규약 불일치 | `retarget.py` + `rig_tpose.json` (selftest T12/T13) |

## 2026-10-04 상시 실행 대비 (USAGE.md §9)

20분 실시간 soak·장애 주입으로 확인. 단시간 측정으로는 보이지 않던 것들이다.

| 문제 | 수정 |
|---|---|
| `corrector.summary()` 가 누적 리스트 전체를 정렬 → 1시간 분량 29ms / 8시간 352ms / 24시간 1.4초 루프 정지(5초마다), 통계 ≈14MB/시간 | 분위수는 최근 9000프레임, 나머지는 누적값 (`_Series`). 24시간 분량 1.8ms. 창 이하 길이에서는 예전 숫자와 동일 |
| 송신 `OSError` 가 try 밖 → 브리지 사망 | `Bridge.step` 이 세고 넘어간다 (`send_errors`) |
| torch 기본 6스레드가 프레임 사이 spin → **5.11코어 상시** | `--threads` (서버 기본 1) → 0.42코어. 보정 결과 동일, p95 9.8→20.1ms |
| 크래시·멈춤 시 복구 없음, 로그 없음, 에폭 미고정, 중복 실행 | `server.py` supervisor (장애 주입 10/10) |
| 콘솔 통계의 예산초과 %가 누적값이라 오래 돌면 최근 스파이크를 묻음 | 최근 창 기준으로 표시 |

## 2026-10-08 수정 사항 (루프백 자체 검사 — 데이터셋 불필요)

| 변경 | 내용 |
|---|---|
| 입력 모션 | held-out 데이터셋 첫 파일 → `vmc_bridge.synthetic_penetrating_motion()`: VRM T-포즈에서 양팔을 내려 팔꿈치를 접고(아래팔이 몸통을 뚫고 교차) 다리를 4° 모은 자세를 천천히 흔든 90프레임. **모든 프레임에서 4쌍 전부 관통 2.0~5.2cm** |
| 새 검사 | 입력 관통 ≥ 1cm(공허 방지) / 수신측에서 잰 보정 프레임의 4쌍 관통 ≤ 0.05cm / 보정 예외 0건(`Bridge.step` 이 예외를 원본 통과로 흡수하므로 세지 않으면 통과해 버렸다) |
| 효과 | 데이터셋 없는 배포판(Release)에서도 `--loopback-test` 가 돈다. 깨끗한 입력과 달리 사영의 그래디언트 경로를 실제로 탄다 |

검증: 원본·Release 모두 통과(보정 프레임 61개 관통 max 0.000cm), selftest 16/16(T15 가 이 함수를 쓴다). 일부러 고장 낸 5가지 — 사영 끔 / 정규화 yaw 안 되돌림 / 사영 예외 / 사영을 inference_mode 로 / 입력 관통 기준 10cm — 전부 실패로 잡힘. (옛 검사도 사영 예외에서 실패는 했지만 원인이 아니라 'Hips 위치 불일치'라는 간접 증상으로였다.) 롤백: 수정 전 `vmc_bridge.py` 사본으로 되돌린다.

## 2026-10-07 수정 사항 (보정 아바타 튐 — `claude_analysis/streaming_jitter_report_20261006.md`)

| 변경 | 내용 | 롤백 |
|---|---|---|
| 출력 자리 `emit_lag` 0 → **2** | 윈도우 끝(미래 문맥 없음) 대신 2프레임 앞을 내보낸다. 지연 +67ms | `server_config.json` `"emit_lag": 0` / `--emit-lag 0` |
| hold 틱 미주입 | 새 트래커 프레임이 없는 틱은 모델에 넣지도 보내지도 않는다. 그 사이 온 표정·손가락은 직전 출력에 실어 보낸다 | `"hold_repush": true` / `--hold-repush` |
| loopback 검사 | 출력이 delay 프레임 늦은 것을 반영 | — |

실측 (tracker_sim persistent → UDP → 브리지(server.py 명령줄) → 기록기, 60초, 파일 경계 제외; 중간점 이탈 = 관절 최대 cm):

| 조건 | B p99 | B >6cm /1000프레임 | 원본 C >6cm | 손상 입력 A >6cm |
|---|---:|---:|---:|---:|
| 30fps 이전 / 이후 | 15.2 → **9.7** | 48 → **25** | 4 | 36 |
| 25fps 이전 / 이후 | 15.2 → **9.1** | 46 → **22** | 3 | 42 |
| clean 30fps 이전 / 이후 | 13.5 → **9.7** | 35 → **23** | 4 | 4 |

- 개선은 거의 전부 `emit_lag` 몫이다. hold 미주입만(lag 0)은 30fps 196 vs 212(>3cm), lag 2 위에서는 측정 오차 안.
- **남은 것**: clean 입력에서도 B 의 큰 튐(>6cm)이 원본의 약 6배 — do-no-harm 문제(재학습 영역). 전환 팝(첫 1초 뒤 1회)은 그대로.

## 2026-10-03 수정 사항

| 문제 | 증상 | 수정 | 검증 |
|---|---|---|---|
| `stride>1`·`emit_lag>0` 출력 시간축 | 워밍업 때 이미 내보낸 프레임을 첫 계산이 다시 큐에 넣어 **출력이 뒤로 점프** (stride=30: …26,27,28 → 0,1,2…, 이후 29프레임 지연 / lag=2: 28 → 27). `run_sim --stride 30` 의 clean 이동량이 **140.85cm** 로 나오는 등 측정이 통째로 오염 | `corrector.py` — 큐 항목에 절대 프레임 번호를 붙이고, push 번호 t 에서 항상 입력 t−delay 를 내보낸다 (`delay = stride−1+emit_lag`, `_emit()`) | 기본 설정(1/0) 출력 **비트 동일**(3시나리오 + bypass 토글), selftest **T16** 신설, 수정 후 stride=30 clean 이동량 0.67cm |
| 브리지 끊김 리셋 시각 | 끊김을 '무신호 판정 시각'부터 재서 실제로는 **2.5초**에야 리셋 (문서는 2초) | `vmc_bridge.py` — 끊김 시작 = 마지막 수신 시각 | 코드 검토 (실 트래커 끊김은 미재현) |
| 브리지가 싱크 설정을 덮어씀 | 변환을 우회하려고 `sink.retarget` 을 영구히 바꿔치기 → `--no-retarget` 에서도 보고가 "변환 ON" | `sinks.VmcOscSink.send(..., already_vrm=True)` 인자로 우회, 바꿔치기 코드 삭제 | selftest T14/T15 |
| T15 가 정규화 회전을 검사하지 않음 | 데이터셋 첫 프레임이 이미 정면이라 세션 yaw 0.0° — 회전 경로가 공허하게 통과 | `loopback_test(yaw_deg=37)` — 트래커를 돌려 보내고 모델 입력이 원래 데이터셋 프레임인지 검사 | 37°/0°/−120° 통과(쿼터니언 오차 1.2e-7), **정규화를 고장 내면 실패** 확인 |

## 실측 결과 (`l10.1_phys0.5` epoch 4000 = 현재 기본 로드, CPU, 600프레임, held-out 8파일, 2026-10-03)

`python AI_Streaming/run_sim.py --scenario <시나리오>` 그대로(기본 옵션: stride 1, emit_lag 0, 예산 가드 ON).

| 시나리오 | 지연 p50/p95/max (ms) | 예산 초과 | 보정 프레임의 4쌍 관통 | 입력 대비 이동 (전신/손) | 클린 대비 오차 (입력→출력) |
|---|---|---|---|---|---|
| clean | 5.6 / 9.2 / 23.7 | 0.00% | 0.000 cm | **1.58 / 1.38 cm** | 0.00 → 1.58 cm |
| transient | 6.7 / 8.1 / 19.9 | 0.00% | 0.000 cm | 1.75 / 2.00 cm | 0.52 → 1.84 cm |
| persistent | 5.8 / 14.1 / 20.5 | 0.00% | **0.000 cm** (입력 max 8.54) | 2.54 / 10.82 cm | 1.56 → 1.45 cm |
| legacy80 (OOD) | 5.5 / 6.8 / 13.0 | 0.00% | 0.000 cm (입력 max 7.72) | 4.16 / **23.58 cm** | 4.04 → 1.61 cm |

* 지연은 실행마다 흔들린다(persistent 재실행 p95 15.0 / max 26.3ms, clean 재실행 p95 10.9 / max 17.9ms).
  관통·이동량·오차는 재실행에서 **비트 단위로 같았다**(결정론).
* "보정 프레임" = 출처가 corrected 인 571프레임. 스트림 첫 29프레임은 워밍업 원본 통과라 따로 센다
  (persistent 의 전체 출력 max 1.847cm 는 전부 워밍업 구간의 원본이다).

### epoch 100 대비 (같은 명령, `--epoch 100`)

| 항목 | epoch 100 | epoch 4000 |
|---|---|---|
| clean 이동 (전신/손) — do-no-harm | 3.20 / 8.21 cm | **1.58 / 1.38 cm** |
| persistent 보정 프레임 관통 | max 0.941 cm, 2.80% 프레임 | **0.000 cm, 0%** |
| persistent 지연 p95 / max, 예산 초과 | 28.5 / 40.8 ms, 0.50% | **14.1 / 20.5 ms, 0%** |
| persistent 사영 반복 최대 | 4회 | 2회 |
| persistent 클린 대비 오차 | 1.56 → 3.55 cm (악화) | 1.56 → **1.45 cm (개선)** |
| legacy80 손 이동 | 25.49 cm | 23.58 cm |
| legacy80 클린 대비 오차 | 4.04 → 3.41 cm | 4.04 → **1.61 cm** |

epoch 100 의 수치는 예전 이 문서의 표(3.20/8.21, 0.941cm)를 그대로 재현한다 — 측정 경로는 변하지 않았고
**모델이 바뀌어서 숫자가 바뀐 것**이다.

### 이 표에서 읽어야 할 것

**1. "예산과 보장은 동시에 못 가진다"는 epoch 100 의 결론이었다 — epoch 4000 에서는 이 표본에서 재현되지 않는다.**

| persistent | epoch 100 | epoch 4000 |
|---|---|---|
| 예산 가드 ON | p95 28.5 / max 38.6~40.8 ms, 초과 0.5~0.8%, **잔여 관통 0.941 cm** | p95 14.1 / max 20.5 ms, 초과 0%, 잔여 0.000 cm |
| 예산 가드 OFF (`--no-adaptive-k`) | p95 38.3 / max 74.2 ms, 초과 7.00%, 잔여 0.000 cm (2026-09 측정, 이번엔 미재측정) | p95 14.2 / max 36.1 ms, 초과 0.17%(1프레임), 잔여 0.000 cm |

epoch 4000 은 모델 단계에서 관통을 더 많이 지워 사영이 최대 2회 반복으로 끝난다. 그래서 가드가 K 를
211번 깎아도(24 → 예산 안의 값) 필요한 2회보다 많이 남아 **가드 ON/OFF 의 출력이 완전히 같다**.
단, 이것은 held-out 8파일·600프레임 한 표본의 결과다. 사영 비용 자체에 상한이 생긴 것은 아니므로
(가드 OFF 에서 단발 36ms) **지연 상한 '보장'은 여전히 없다** — 가드는 계속 켜 둔다.

**2. do-no-harm 은 크게 나아졌지만 0 은 아니다.** clean 입력에서 전신 1.58cm / 손 1.38cm 를 움직인다
(epoch 100 은 3.20 / 8.21cm). transient 는 입력 대부분이 멀쩡해서(충돌 프레임 7.5%) 클린 대비 오차가
0.52 → 1.84cm 로 오히려 늘어난다 — 이 이동량이 그대로 비용으로 잡히는 것이다.

**3. legacy80(학습 분포 밖)에서 손이 23.6cm 움직이는 것은 여전히 교정이 아니라 동작 삭제에 가깝다.**
클린 대비 오차는 1.61cm 로 줄었지만(팔을 원래 자리 쪽으로 되돌림) 입력 의도와는 크게 다르다. 재학습 문제다.

## 남아 있는 한계 (런타임에서 해결 불가)

- do-no-harm 이 0 이 아님 (epoch 4000 clean 1.58cm — 재학습/손실 설계 문제)
- 4쌍 밖 전신 관통 4~6cm 잔존 (2026-09 측정, epoch 4000 미재측정)
- 지연 상한의 **보장** 없음 (epoch 4000 표본에서는 예산 초과 0% 였지만 사영 비용은 입력 의존)
- 프리즈/텔레포트/트래커 소실은 학습에서 본 적이 없는 손상 유형
- 리그 T-포즈의 뼈축 비틀림(롤)은 기하로 확정 불가 — Warudo 에서 눈으로 확인
- 트래커가 30fps 보다 느리면 새 프레임이 있는 틱에만 보정·송신한다(2026-10-07 hold 미주입, 보간 없음). `--hold-repush` 면 예전처럼 같은 프레임이 링버퍼에 반복해 들어간다
