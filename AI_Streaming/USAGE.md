# AI_Streaming 사용법 상세

README 가 "무엇이고 왜 이렇게 만들었나"라면 이 문서는 **"어떻게 쓰나"** 다.
모든 명령은 **MCC 루트**(`C:\Users\Contemplator\Desktop\MCC`)에서 실행하는 것을 기준으로 적었다.
(`paths.py` 가 절대 경로를 쓰므로 다른 곳에서 실행해도 같은 파일을 보지만, 상대 경로로 준
`--log-csv` 같은 출력 파일은 실행 위치에 생긴다.)

- 필요 패키지: `torch`, `python-osc`(VMC 송수신 — `vmc_bridge`·`tracker_sim`·selftest T15).
- 콘솔 한글이 깨지면 PowerShell 에서 `$env:PYTHONIOENCODING="utf-8"` 를 먼저 둔다.
- 장치는 **CPU 권장**. 한 번에 30프레임 윈도우 하나만 도는 구조라 CUDA 는 오히려 3~6배 느렸다(실측).

---

## 0. 처음 쓸 때의 순서

| 단계 | 명령 | 통과 기준 |
|---|---|---|
| 1 자체 점검 | `python AI_Streaming/selftest.py` | `결과: 16/16 통과` |
| 2 품질·지연 | `python AI_Streaming/run_sim.py --scenario persistent` | 보정 프레임 관통 0.000cm, 예산 초과 ≈0% |
| 3 소켓 왕복·관통 제거 | `python AI_Streaming/vmc_bridge.py --loopback-test` | `[loopback] 관통 제거 … 관통 max 0.000 cm` + `[loopback] OK` |
| 4 Warudo 눈 확인 | `python AI_Streaming/vmc_bridge.py --replay clean` | 아바타가 뒤틀리지 않음, 팔 롤·손바닥·발 기울기·골반 높이 정상 |
| 5 가상 장비 실측 | 창 2개: 브리지 + `tracker_sim.py` (§5, Warudo 비교는 §5.1) | 콘솔 p95 < 33ms, 브리지 `관통 … 입력 → 출력 0.00`, 캐릭터 A/B/C 비교 |
| 6 라이브 | `python AI_Streaming/vmc_bridge.py --listen-port 39540` + 실제 트래커 | — |

`rig_tpose.json` 은 이미 만들어져 있다. 데이터셋(`processed_motions_VMC`)이 바뀌었을 때만
`derive_rest_pose.py` 를 다시 돌린다(§6).

---

## 1. 어떤 모델이 로드되나 — 가장 먼저 알아야 할 것

모든 실행 스크립트는 `--run`(체크포인트 폴더 이름)과 `--epoch` 를 받는다.

| 인자 | 기본값 | 의미 |
|---|---|---|
| `--run` | `tfm_declip_cov_l10.1_anat_recon1_phys0.5_kl0` (`paths.DEFAULT_RUN`) | `checkpoints/` 아래 폴더 이름, 또는 폴더 경로 |
| `--epoch` | 없음 = **폴더 안의 가장 큰 에폭** | `pvtvae_epoch_<N>.pth` 의 N |

⚠️ 기본 폴더에는 지금 epoch 4000 까지 있어 **epoch 4000 이 로드된다**. 학습을 이어 하면 아무 경고 없이
로드 대상이 바뀐다. 측정을 재현하거나 비교할 때는 반드시 `--epoch` 로 고정한다.
`selftest.py` 는 인자가 없고 항상 기본 런의 최대 에폭을 쓴다.
사용 가능한 런 목록은 `python -c "import sys; sys.path.insert(0,'AI_Streaming'); import paths; print(paths.available_runs())"`.

---

## 2. `selftest.py` — 출시 전 자체 점검 (인자 없음)

```bash
python AI_Streaming/selftest.py
```

약 1분. 실패가 하나라도 있으면 종료 코드 1. 항목:

| 묶음 | 항목 | 무엇을 막는가 |
|---|---|---|
| [A] 계약 | T1 학습 코드와 본 순서·충돌 4쌍 일치 / T2 pack·unpack 왕복 / T3 계약 검사기가 오배선을 잡는가 | 텐서 규약이 조용히 어긋나는 것 |
| [B] 런타임 함정 | T4 `inference_mode` 가드 / T5 워밍업 29프레임 원본 통과 / **T6 실제 관통을 사영으로 제거** / T7 결정론 | 방송 중 첫 자기충돌에서 터지는 결함 |
| [C] 순서 | T8 필터→사영 순서를 뒤집으면 관통이 되살아남 | 파이프라인 순서 변경 |
| [D] 한계 기록 | T9 29프레임 입력은 에러 없이 다른 답 / T10 31프레임은 RuntimeError | (실패가 아니라 기록) |
| [E] Warudo | T11 Unity 본 이름 / T12 리그 T-포즈 → VRM 단위 회전 / T13 리타깃이 관절 위치 보존 / T14 VMC 패킷 형식 / **T15 UDP 루프백**(트래커 yaw 37° 정규화 + 합성 관통 모션 제거 포함) | Warudo 에서 아바타 뒤틀림·이중 전송 |
| [F] 시간축 | **T16** stride/emit_lag 를 바꿔도 출력 프레임 번호가 0,1,2,… 연속 | 출력이 과거로 점프하는 것 |

T15 는 로컬 UDP 포트 **39565/39566** 을 쓴다. Warudo 나 다른 프로그램이 그 포트를 잡고 있으면 실패한다.
검사들은 "조건을 만족하는 윈도우를 찾아서" 쓰고, 못 찾으면 공허한 통과 대신 **실패**로 처리한다.

---

## 3. `run_sim.py` — 라이브 시뮬레이션 (송신 없음)

held-out 모션을 한 프레임씩 보정기에 밀어 넣고 **지연 · 관통 · do-no-harm** 을 한 번에 잰다.

```bash
python AI_Streaming/run_sim.py                                   # clean, 600프레임
python AI_Streaming/run_sim.py --scenario persistent --frames 900
python AI_Streaming/run_sim.py --scenario persistent --no-adaptive-k   # 예산 가드 끄고 대조
python AI_Streaming/run_sim.py --scenario legacy80 --no-projection     # 사영 끄고 대조
python AI_Streaming/run_sim.py --epoch 100                       # 옛 체크포인트와 비교
python AI_Streaming/run_sim.py --stride 30                       # 타일링 (30프레임마다 1회 계산)
python AI_Streaming/run_sim.py --emit-lag 2                      # 2프레임 늦게 내보내기
python AI_Streaming/run_sim.py --pace --osc-dry-run              # 실제 30fps 시계 + VMC 패킷 생성
python AI_Streaming/run_sim.py --large --run <AI_model_large 런 폴더>
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--scenario` | clean | `clean` 원본 그대로(do-no-harm) / `transient` 창 안에서 들어왔다 나가는 손상 / `persistent` 창 전체 지속 손상 / `legacy80` 고정 80° 팔 주입(학습 분포 밖) |
| `--frames` | 600 | 흘릴 프레임 수 (30fps 기준 20초) |
| `--files` | 8 | held-out(test 스플릿) 파일 수. 파일이 짧으면 다음 파일로 넘어간다 |
| `--device` | cpu | |
| `--stride` | 1 | 몇 프레임마다 계산할지. 1 = 매 프레임(기본·라이브), 30 = 타일링 |
| `--emit-lag` | 0 | 윈도우 끝에서 몇 프레임 앞을 내보낼지. 2 면 영위상 필터가 양쪽 이웃을 다 가진 프레임 |
| `--budget-ms` | 33.33 | 프레임당 시간 예산. 예산 가드가 이 값으로 사영 반복 K 를 깎는다 |
| `--no-lowpass` / `--no-projection` | 끔 | 단계를 빼고 대조 |
| `--no-adaptive-k` | 끔 | 예산 가드를 끄고 K=PROJ_K(24) 고정 |
| `--pace` | 끔 | 실제 시계로 30fps 페이싱 (기본은 최대 속도) |
| `--validate-every` | 90 | N프레임마다 입력 계약 검사 |
| `--osc-dry-run` | 끔 | VMC 패킷을 만들기만 하고 보내지 않음 → 마지막에 첫 패킷 예시 출력 |
| `--large` | 끔 | `AI_model_large` 의 확장 아키텍처로 로드. **`--run` 도 반드시 그 런 폴더로** 줘야 한다(기본 런은 기본 아키텍처라 로드 실패) |

**출력 지연**: `stride-1+emit_lag` 프레임. 예) stride 30 → 29프레임(≈967ms), emit_lag 2 → 2프레임(67ms).
그만큼 스트림 시작에서 출력이 없고, 이후 출력 k 번째는 항상 입력 k 번째와 짝이 맞는다.

### 출력 읽는 법

```
[1] 지연      p50/p95/max ms, 예산 초과 %, 단계별 평균(모델/필터/사영), 사영 반복 최대·K 깎인 횟수,
              워밍업 통과 프레임, 출력 지연
[2] 관통      입력/출력의 4쌍 관통 max·충돌 프레임 %, 최악 관통 제거율
              └ 원본 통과분(워밍업/패닉)   ← 파이프라인이 손대지 않은 구간
              └ 보정된 프레임             ← 파이프라인의 실제 성적 (이 줄로 판정)
[3] 충실도    입력 대비 이동량(전 관절/손), 클린 대비 오차(입력→출력), 지터
[4] 계약      입력 계약 위반 검출 건수, 프레임 in/out
[5] 송신      (--osc-dry-run 일 때) VmcOscSink 보고
```

- 판정은 평균이 아니라 **p95 / max** 로 한다. 지연은 실행마다 흔들리고(같은 명령 재실행 p95 14.1 vs 15.0ms),
  관통·이동량은 결정론이라 재실행해도 같다.
- **clean 의 "입력 대비 이동량" 이 do-no-harm 지표**다. 0 에 가까울수록 좋다.
- `[2]` 의 "출력 max" 는 워밍업 원본을 포함한다. 보정 실패 여부는 "보정된 프레임" 줄로만 본다.
- 지터는 연속 스트림 기준이라 `docs/evaluate_results.csv` 의 행과 직접 비교할 수 없다.

---

## 4. `vmc_bridge.py` — 실제 송수신 (트래커 → 보정 → Warudo)

```
트래커 ─VMC/UDP─▶ [--listen-port] 수신 → VRM→리그 변환 + 세션 정규화 → 보정 → 되돌림 → [--send-port] ─▶ Warudo
```

### 실행 형태

**VSCode "Run Python File"** (train.py 와 같은 규칙): `vmc_bridge.py` 위쪽 `[실행 설정]` 블록의
`MODE`("live" / "replay" / "loopback")와 포트·옵션 상수를 바꾸고 인자 없이 실행한다. 실행하면 같은
설정의 명령줄이 `[실행 설정] python ...` 로 출력된다. 인자를 하나라도 주면 그 블록은 무시된다(server.py 영향 없음).

```bash
python AI_Streaming/vmc_bridge.py --loopback-test                       # Warudo·데이터셋 없이 자체 검사 (포트 39555/39556)
python AI_Streaming/vmc_bridge.py --replay clean                        # 데이터셋 모션을 Warudo(39539)로 재생
python AI_Streaming/vmc_bridge.py --replay persistent --replay-hips-height 1.0
python AI_Streaming/vmc_bridge.py --listen-port 39540 --send-port 39539 # 라이브
python AI_Streaming/vmc_bridge.py --listen-port 39540 --dry-run         # 수신·보정만, 전송 안 함
python AI_Streaming/vmc_bridge.py --listen-port 39540 --log-csv bridge_log.csv
python AI_Streaming/vmc_bridge.py --replay clean --no-retarget          # [시연] 변환 없이 → 뒤틀림
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--run` / `--epoch` | §1 | 로드할 체크포인트 |
| `--listen-port` | 39540 | 트래커가 보내는 포트. **트래커 앱의 송신 대상을 이 포트로** 바꾼다 |
| `--send-host` / `--send-port` | 127.0.0.1 / 39539 | Warudo VMC Receiver 주소 (Warudo 기본 39539) |
| `--fps` | 30 | 브리지 시계. 트래커가 더 빠르면 최신 프레임만 쓰고, 새 프레임이 없는 틱(hold)은 모델에 넣지도 보내지도 않는다(보간 없음, 2026-10-07) |
| `--emit-lag` | **2** | 윈도우 끝에서 몇 프레임 앞을 내보낼지. 2 = +67ms 지연으로 끝자리 튐 완화, 0 = 2026-10-07 이전 동작(지연 최소, 튐 큼) |
| `--hold-repush` | 끔 | hold 틱에도 같은 프레임을 모델에 다시 넣는다 — 2026-10-07 이전 동작(비교·롤백용) |
| `--dry-run` | 끔 | 송신 안 함 |
| `--replay` | 없음 | 트래커 대신 held-out 데이터셋을 재생 (같은 프로세스 안 — UDP 수신 경로는 안 탄다) |
| `--replay-hips-height` | 0.9 | 재생 시 골반 높이(m). 아바타가 뜨거나 잠기면 조정 |
| `--no-fold-upperchest` | 끔 | 트래커의 UpperChest 를 Chest 에 접지 않는다 |
| `--no-lowpass` / `--no-projection` / `--no-adaptive-k` | 끔 | run_sim 과 같음 |
| `--no-retarget` | 끔 | [디버그] 리그↔VRM 변환을 끈다. 아바타가 뒤틀리는 것이 정상 |
| `--loopback-test` | 끔 | 합성 트래커(yaw 37°) → 브리지 → 로컬 수신기. **데이터셋 불필요** — 매 프레임 4쌍이 관통하는 합성 모션(2.0~5.2cm)을 보내 UDP·번들 파싱·Hips 절대 위치 복원·정규화·pass-through·**수신된 보정 프레임의 관통 0**·보정 예외 0 을 검사 (포트 39555/39556) |
| `--stats-every` | 5.0 | 콘솔 통계 주기(초) |
| `--idle-after` | 0.5 | 트래커 수신이 이만큼 없으면 송신 정지 (Warudo 는 마지막 포즈 유지) |
| `--reset-after` | 2.0 | 이만큼 이상 끊겼다 돌아오면 세션 리셋(정규화 기준·링버퍼) → 재개 첫 1초는 원본 통과 |
| `--log-csv` | 없음 | 프레임별 기록 CSV |

실행 중 키: **`b`** = bypass 토글(원본 통과, 패닉 스위치), **`q`** = 종료. (Windows 콘솔에서만 — `msvcrt`)

### 콘솔 통계 줄 (5초마다, 아래 숫자는 형식 예시)

```
in 1520 out 1520 | 처리 p95  14.1 ms max  20.5 | 예산초과 0.0% | K깎임 211 | hold 0 | 오류 0 | bypass=False | 트래커 1520f
    관통 4쌍 max 입력  8.62 → 출력  0.00 cm  [최근 5s]
```

| 항목 | 의미 |
|---|---|
| in / out | 브리지가 처리한 프레임 / Warudo 로 보낸 프레임 |
| 처리 p95 / max | 수신 1프레임의 변환+보정+송신 시간 |
| 예산초과 | 33.3ms 를 넘은 프레임 % |
| K깎임 | 예산 가드가 사영 반복 상한을 줄인 횟수 (잔여 관통이 0 이면 무해) |
| hold | 새 트래커 프레임이 없던 틱 수 (트래커가 30fps 미만이거나 브리지가 밀리면 늘어남). 기본은 이 틱을 모델에 넣지 않는다 — `--hold-repush` 일 때만 같은 프레임을 다시 넣는다 |
| 오류 | 보정 예외 → 원본 통과한 횟수 (처음 3번은 메시지 출력) |
| 트래커 Nf | `/VMC/Ext/T` 로 센 트래커 프레임 수 |
| 관통 4쌍 max 입력 → 출력 | 이번 5초 동안 브리지가 **받은** 프레임과 Warudo 로 **보낸** 프레임의 4쌍 관통 최댓값(cm). 보낸 프레임을 FK 로 다시 잰 값이라 보정이 실제로 됐는지 바로 보인다. 출력은 보정본만 센다 — 워밍업(시작·재개 첫 1초)·bypass·오류로 원본을 통과시킨 프레임은 `(원본통과 Nf max …)` 로 따로 붙는다. 측정(≈2.7ms/틱)은 처리 ms 에 넣지 않는다. 종료 시 `세션 …` 줄에 전체 max |

### `--log-csv` 열

`t_s, frame_in, emitted_stage(warmup/corrected/bypass/passthrough/error/hold_skip), ms, k_used, k_capped,
proj_residual_cm, proj_frames_touched, hold, error, pen_in_cm, pen_out_cm` — 행마다 flush 되므로 강제 종료해도 남는다.
`hold_skip` 행은 새 트래커 프레임이 없어 모델에 넣지 않은 틱이다. `pen_in_cm`/`pen_out_cm` = 그 틱에 받은/보낸 프레임의
4쌍 관통 max (보낸 것이 없으면 빈칸; 출력은 입력보다 emit_lag 프레임 늦은 프레임이다).

### 무엇이 통과되고 무엇이 바뀌나

| VMC 메시지 | 처리 |
|---|---|
| `/VMC/Ext/Bone/Pos` 21본 (Hips·척추·팔·다리·머리) | **보정 대상** |
| `/VMC/Ext/Bone/Pos` 그 밖 (손가락·눈·Jaw·UpperChest) | 그대로 통과 |
| UpperChest | Chest 에 접어 모델에 넣고, 출력에서 풀어 원본 UpperChest 를 통과 |
| `/VMC/Ext/Blend/*` (표정) | 그대로 통과 |
| `/VMC/Ext/Root/Pos` | 트래커 값 그대로 통과 |
| `/VMC/Ext/OK` | 버리고 브리지 것(`1 3 0 1`)을 보냄 |
| `/VMC/Ext/T` | 프레임 경계로 세고, 브리지 시계로 새로 보냄 |

21본 중 하나라도 아직 안 왔으면 보정을 시작하지 않고 `수신 대기… (빠진 본 [...])` 를 출력한다.

### Warudo 쪽 설정

- 캐릭터 → 모션 캡처 → **VMC Receiver**, 포트 39539, **VRM 모델**(정규화 본, T-포즈에서 본 회전 0).
- 트래커가 Warudo 로 직접 보내면서 브리지도 보내면 두 스트림이 겹쳐 아바타가 떤다 → 트래커는 브리지로만.
- Warudo 의 VMC Sender 를 입력으로 쓰면서 같은 캐릭터가 수신하면 **되먹임 고리** → 캐릭터 분리.

---

## 5. `tracker_sim.py` — 가상 모션캡처 장비

데이터셋을 **별도 프로세스·자기 시계·UDP** 로 보내 브리지와 Warudo 가 실제 장비와 구분하지 못하게 한다.

**VSCode "Run Python File"**: `tracker_sim.py` 위쪽 `[실행 설정]` 블록의 기본값이 아래 창 2 명령과 같은
3-아바타 비교(persistent, 반복, 39540·39541 + 원본 39542)다. 브리지(`vmc_bridge.py`, `MODE="live"`)를 먼저
▶ 로 띄운 뒤 이 파일을 ▶ 로 실행한다. 인자를 하나라도 주면 그 블록은 무시된다.

```bash
# 창 1 — 브리지 ("수신 대기…" 가 뜨면 준비 완료)
python AI_Streaming/vmc_bridge.py --listen-port 39540 --send-port 39539 --log-csv bridge_log.csv

# 창 2 — 가상 트래커: 손상 입력을 브리지(39540)와 캐릭터 A(39541)로, 정답을 캐릭터 C(39542)로
python AI_Streaming/tracker_sim.py --scenario persistent --loop \
    --to 127.0.0.1:39540 --to 127.0.0.1:39541 --clean-to 127.0.0.1:39542 --log-csv tracker_log.csv
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--scenario` | clean | run_sim 과 같은 4종 |
| `--to host:port` | 127.0.0.1:39540 | 손상 입력 송신 대상. **반복 가능** |
| `--clean-to host:port` | 없음 | 손상 주입 전 원본(정답) 송신 대상. 반복 가능 |
| `--fps` | 30 | 송신 프레임률. 60 이면 브리지의 sample-and-hold(절반 버림) 검사 |
| `--files` | 8 | held-out 파일 수 |
| `--seconds` | 0 | 송신 시간. 0 = 파일 끝까지 (`--loop` 면 무한) |
| `--loop` | 끔 | 파일 끝에서 반복 |
| `--hips-height` | 0.9 | 골반 높이(m). 데이터셋 Hips 는 첫 프레임 기준 상대값이라 이게 없으면 바닥에 묻힌다 |
| `--seed` | 20260707 | 손상 주입 시드 (같은 시드 = 같은 손상) |
| `--no-retarget` | 끔 | [시연] 변환 없이 보내기 |
| `--log-csv` | 없음 | `t_s, frame, pen_in_cm, pen_clean_cm` (입력·정답의 4쌍 관통) |

`q` 로 종료. 5초마다 `보냄 Nf / 실측 fps / 입력(보정 전) 4쌍 관통 max 최근·누적 / 충돌 프레임 %` 를 출력한다.
이 관통은 브리지로 보내기 **전**의 손상 입력이라 보정과 무관하게 줄지 않는다 — 보정 결과는 브리지 콘솔의
`관통 4쌍 max 입력 … → 출력 …` 줄에서 본다. (최근 = 이번 5초, 누적 = 시작부터. 2026-10-09 전에는 누적만 찍어
persistent 기본 설정에서 8.62cm 로 고정돼 보였다.)
두 CSV 를 `t_s` 로 맞추면 "이 관통이 들어왔을 때 보정에 몇 ms 걸렸고 잔여가 얼마였나" 를 볼 수 있다.

### 5.1 Warudo 3-아바타 비교 — persistent(손상) / 보정 / 원본

같은 순간의 **손상 입력 · 브리지 보정 출력 · 손상 전 원본** 을 Warudo 캐릭터 3개에 나란히 띄워 눈으로 비교한다.
가상 트래커 하나가 같은 프레임을 세 곳으로 나눠 보내므로 세 캐릭터는 같은 모션을 같은 시계로 움직인다.

```
                     ┌─(손상)──────────────▶ 39541  캐릭터 A  persistent (손상 입력 그대로)
tracker_sim ─────────┼─(손상)──▶ 브리지 39540 ──(보정)──▶ 39539  캐릭터 B  보정 출력
                     └─(원본)──────────────▶ 39542  캐릭터 C  원본 (정답)
```

| 캐릭터 | Warudo VMC Receiver 포트 | 보내는 쪽 | 보이는 것 |
|---|---|---|---|
| A | 39541 | tracker_sim `--to` (두 번째) | persistent 손상이 들어간 입력 — 팔이 몸통을 뚫는다 |
| B | 39539 | vmc_bridge `--send-port` | 보정 결과 — 관통이 사라져야 한다 |
| C | 39542 | tracker_sim `--clean-to` | 손상 주입 전 원본 — B 가 닮아야 할 정답 |

#### 1) Warudo 준비 (한 번만)

1. 같은 VRM 캐릭터를 3개 두고, 각각 모션 캡처 → **VMC Receiver** 를 위 표의 포트로 설정한다(§4 "Warudo 쪽 설정").
   포트가 서로 겹치면 두 스트림이 한 캐릭터에 섞여 떤다.
2. 캐릭터를 옆으로 나란히 놓는다. 세 스트림 모두 Hips 높이 0.9 m·같은 위치로 보내므로, 캐릭터 위치를 옮겨도
   겹쳐 보이면 그 사실을 기록해 둔다(Warudo 가 Hips 위치를 캐릭터 기준으로 적용하는지 아직 확인하지 않았다).

#### 2) 실행 순서 — 브리지 먼저

| 순서 | VSCode ▶ (인자 없이 실행) | 터미널에서 같은 동작 |
|---|---|---|
| ① 브리지 | `vmc_bridge.py` — `[실행 설정]` 의 `MODE = "live"` (기본) | `python AI_Streaming/vmc_bridge.py --listen-port 39540 --send-port 39539` |
| ② 가상 트래커 | `tracker_sim.py` — `[실행 설정]` 기본값이 곧 이 구성 (`SCENARIO = "persistent"`, `TO` 39540·39541, `CLEAN_TO` 39542, `LOOP`) | `python AI_Streaming/tracker_sim.py --scenario persistent --loop --to 127.0.0.1:39540 --to 127.0.0.1:39541 --clean-to 127.0.0.1:39542` |

- 브리지 창에 `수신 대기… (메시지 0개 …)` 가 뜬 뒤 ②를 실행한다.
- **`MODE = "replay"` 로 두면 안 된다** — replay 는 브리지 안에서 데이터셋을 직접 재생하고 39540 수신을 하지 않으므로
  B 가 A·C 와 다른 모션을 하게 된다.
- 종료: 두 창에서 각각 `q`. tracker_sim 만 멈추면 브리지는 0.5초 뒤 송신을 멈추고(B 는 마지막 포즈 유지),
  2초 이상 지나 다시 ②를 실행하면 세션을 리셋한다 → 재시작 직후 1초는 B 가 A 와 같다(워밍업, 정상).

#### 3) 콘솔에서 확인할 것

```
[브리지]       in 1325 out 1323 | 처리 p95  14.3 ms ... | 오류 0 | bypass=False | 트래커 1341f
                   관통 4쌍 max 입력  8.62 → 출력  0.00 cm  [최근 5s]
[tracker_sim]  보냄   1340f  실측  29.4 fps  | 입력(보정 전) 4쌍 관통 max 최근  8.62 / 누적  8.62 cm ...
```

- **보정이 됐는지는 브리지의 `입력 → 출력` 줄로 본다.** 출력(= 캐릭터 B 에 보낸 프레임)이 0.00 이면 관통이 없다.
- tracker_sim 의 관통 숫자는 **캐릭터 A 의 값(보정 전)** 이라 보정과 무관하게 줄지 않는다. `누적` 은 한 번 깊은
  관통이 지나가면 그 값에 머문다 — 지금 들어오는 관통은 `최근` 을 본다.
- 처리 p95 가 33ms 를 넘거나 `오류` 가 늘면 B 의 화면 결과도 믿지 않는다(오류 프레임은 원본 = A 와 같은 포즈가 나간다).

#### 4) 언제 무엇을 보나 — 기본 설정의 깊은 관통 시각

시드(20260707)·파일 수(8)·평지 펴기가 기본값이면 손상 주입이 매번 같다. 아래는 tracker_sim 시작 시각 기준
첫 바퀴(약 85초)에서 입력 관통이 3cm 를 넘는 주요 구간이다(2026-10-09 실측, 실시간 송신 결과와 5초 창 max 일치).
두 번째 바퀴부터는 손상 난수가 이어져 다른 손상이 들어간다.

| 시각 (s) | 입력 관통 최대 | 쌍 | A 에서 보이는 것 |
|---|---|---|---|
| 1.0 ~ 1.4 | 4.31 cm | 오른아래팔↔몸통 | 브리지 워밍업(첫 1초) 직후 — B 도 이 무렵부터 보정본이 나간다 |
| 6.1 ~ 9.0 | 8.17 / 8.54 cm | 왼아래팔↔몸통 | 왼팔이 몸통 깊이 박힘 (반복 3~4회) |
| 11.0 ~ 13.7 | 5.87 cm | 오른아래팔↔몸통, 아래팔끼리 | |
| 21.4 ~ 21.9 | 4.89 cm | 오른아래팔↔몸통, 아래팔끼리 | |
| 38.0 ~ 39.0 | 4.35 cm | 정강이끼리 | 다리가 서로 파고듦 |
| 41.4 ~ 44.0 | **8.62 cm** (42.1s) | 왼아래팔↔몸통 → 정강이끼리 | tracker_sim `누적` 이 8.62 로 바뀌는 지점 |
| 76.0 ~ 78.9 | **8.93 cm** (77.5s) | 오른 → 왼아래팔↔몸통, 정강이 | 첫 바퀴 최대 |
| 83.0 ~ 85.0 | 4.28 cm | 오른아래팔↔몸통, 정강이끼리 | |

이 구간에서 A 는 팔·다리가 뚫리고, **B 는 뚫리지 않은 채 C 와 비슷한 자세**여야 한다. 관통이 거의 없는 구간
(예: 50~70초, 5초 창 max 0~1cm)에서는 B 와 C 가 거의 같아야 한다 — 여기서 B 가 C 와 눈에 띄게 다르면 관통 제거가 아니라 모델의
충실도 문제다.

#### 5) 비교할 때 알아 둘 것

- **B 는 A·C 보다 약 67ms(2프레임) 늦다** — `EMIT_LAG = 2` 의 의도된 지연이다(§4). 빠른 동작에서 B 가 살짝 뒤따라
  보이는 것은 정상이다. 지연 없이 나란히 보고 싶으면 `EMIT_LAG = 0` (대신 출력이 더 튄다).
- **브리지 창에서 `b`** = bypass 토글. 켜면 B 가 원본 입력을 그대로 받아 A 와 같은 포즈가 된다(지연 2프레임은 그대로) — "보정이 있을 때와 없을 때"를
  같은 캐릭터로 바로 비교할 수 있다. 다시 `b` 로 끈다.
- 다른 손상: `tracker_sim.py` 의 `SCENARIO` 를 `"transient"` / `"legacy80"` / `"clean"` 으로 바꾼다. clean 이면
  A 와 C 가 같고, B 도 같아야 한다(do-no-harm 확인 — 보정이 멀쩡한 모션을 망가뜨리지 않는가).
- 프레임 단위로 남기려면 `LOG_CSV` 를 두 파일 모두에 켠다(브리지 `bridge_log.csv`: `pen_in_cm`·`pen_out_cm`,
  tracker_sim `tracker_log.csv`: `pen_in_cm`·`pen_clean_cm`). `t_s` 는 각자 시작 시각 기준이다.

| 증상 | 원인 |
|---|---|
| B 가 움직이지 않음 | 브리지 `MODE` 가 `"live"` 가 아니거나, B 의 수신 포트가 39539 가 아님 |
| B 가 A·C 와 다른 모션 | 브리지 `MODE = "replay"` (자체 재생) |
| B 와 A 가 똑같음 | bypass 켜짐 / 시작·재시작 직후 1초 워밍업 / 브리지 `오류` 증가 |
| 캐릭터 하나가 떨림 | 두 스트림이 같은 포트로 들어감 (예: tracker_sim `TO` 에 39539 를 넣음) |
| A 와 C 가 어떤 구간은 같고 어떤 구간은 다름 | 정상 — 손상은 30프레임 청크 단위로 주입된다(첫 바퀴 입력 관통 프레임 약 40%) |

---

## 6. `derive_rest_pose.py` — 리그 T-포즈 다시 만들기

```bash
python AI_Streaming/derive_rest_pose.py --check          # 저장하지 않고 검증만 (먼저 이것)
python AI_Streaming/derive_rest_pose.py                  # 도출 + 검증 + rig_tpose.json 덮어쓰기
python AI_Streaming/derive_rest_pose.py --max-files 50 --check   # 빠른 실험
python AI_Streaming/derive_rest_pose.py --out other_tpose.json   # 다른 파일로
```

검증 A~G(팔 수평, 위치 보존, 왕복, 좌우 대칭, Head 중앙값 등)를 매번 출력한다. 실패해도 `--check` 가 아니면
파일은 저장되므로(수치를 보고 판단하라는 뜻), **덮어쓰기 전에 `rig_tpose.json` 을 백업**한다.
Warudo 에서 팔 롤·손바닥·발 기울기가 틀려 보이면 이 파일의 `TPOSE_DIR` / `LATERAL_FIX` 를 조정한다.

---

## 7. Python 에서 직접 쓰기 (API)

모든 모듈은 `AI_Streaming` 을 `sys.path` 에 넣고 import 한다. `paths` 를 먼저 import 하면 `AI_model` 도 경로에 올라간다.

```python
import sys; sys.path.insert(0, r"C:\Users\Contemplator\Desktop\MCC\AI_Streaming")
import paths, contract, sources, sinks, retarget
from corrector import StreamingCorrector
```

### 7.1 보정기 — `StreamingCorrector`

```python
ckpt = paths.checkpoint_path(epoch=4000)          # 또는 paths.checkpoint_path("<런 폴더>", 1500)
corr = StreamingCorrector(ckpt)                   # 생성 시 모델 로드 + 1회 워밍업(약 0.15초)

for frame in my_frames:                           # frame: [87] 텐서 (리그 규약, §7.3)
    out, info = corr.push(frame)
    if out is None:                               # delay>0 일 때 스트림 첫 delay 번
        continue
    send(out)                                     # out = 입력 (현재 번호 - corr.delay) 의 보정본

print(corr.summary())                             # p50/p95/max ms, 예산 초과, K 깎임, 관통 잔여 등
corr.bypass = True                                # 패닉 스위치 (원본 통과, 대기 중인 보정본도 버림)
corr.reset()                                      # 트래커가 끊겼다 붙을 때 (버퍼·큐 비움, 모델·통계 유지)
```

| 생성 인자 | 기본 | 설명 |
|---|---|---|
| `ckpt_path` | (필수) | `.pth` 절대 경로 |
| `device` | "cpu" | |
| `model_class` | `models.TransformerDenoiserCompat` | 확장 아키텍처면 `models_large.TransformerDenoiserLargeCompat` |
| `stride`, `emit_lag` | 1, 0 | `stride+emit_lag ≤ 30`. 지연 `corr.delay = stride-1+emit_lag` |
| `budget_ms` | 33.33 | 예산 가드 기준 |
| `lp_enabled`, `proj_enabled` | True | 저역통과 / 사영 |
| `proj_k` | `projection.PROJ_K` | 사영 반복 상한 |
| `adaptive_k` | True | 예산 가드 |
| `validate_every` | 0 | N>0 이면 N프레임마다 입력 계약 검사 (`info["contract_problems"]`) |
| `physics` | 새로 생성 | 여러 객체가 같은 물리 엔진을 공유하려면 `contract.make_physics()` 를 넘긴다 |

`info` 키: `stage`(이번 push 에서 한 일: warmup/compute/reuse/bypass), `emitted_stage`(내보낸 프레임의 출처:
corrected/warmup/bypass/passthrough), `ms`, `k_used`, `k_capped`, `proj_frames_touched`, `proj_residual_max_cm`, `queue`.

⚠️ `torch.inference_mode()` 안에서 만들거나 돌리면 안 된다(사영이 autograd 를 쓴다) — 생성 시 즉시 막는다.
`torch.no_grad()` 는 괜찮다.

### 7.2 입력원 / 출력부

```python
src = sources.ReplaySource(paths.test_motion_files(limit=8), scenario="persistent",
                           pace=False, loop=False, max_frames=600, seed=20260707)
for inp, clean in src:          # (손상 주입된 입력 [87], 정답 [87])
    ...

rec = sinks.RecordSink();  rec.send(out);  rec.tensor()        # [N, 87] 모아두기
osc = sinks.VmcOscSink()                                       # DRY-RUN (보내지 않음)
osc = sinks.VmcOscSink(client=SimpleUDPClient("127.0.0.1", 39539), hips_offset=(0, 0.9, 0))
osc.send(out)                   # 리그 규약 프레임 → VRM 변환 → OSC 번들 1개
osc.send(vrm_frame, already_vrm=True)   # 이미 VRM 규약이면 변환 건너뜀
print(osc.report())
```

### 7.3 프레임 규약과 검사 — `contract`

87차원 = `[0:3]` Hips 위치(m) + `[3:87]` 21관절 로컬 쿼터니언 `(x,y,z,w)`, 관절 순서는 알파벳순
(`contract.BONE_ORDER`). 30fps, 윈도우 30프레임.

```python
f = contract.pack_frame(hips_xyz_m, {"Hips": (x,y,z,w), "Chest": ..., ...})   # 21본 모두 필요 (빠지면 KeyError)
hips, quats = contract.unpack_frame(f)
problems = contract.validate_frame(f, physics=contract.make_physics())       # [] 이면 정상
contract.check_training_contract()     # 학습 코드와 본 순서·충돌 4쌍 일치 여부
```

### 7.4 리타깃 — `retarget.Retarget`

```python
rt = retarget.Retarget()                 # rig_tpose.json 로드
vrm = rt.rig_to_vrm(frame87)             # 리그 로컬 → VRM 로컬 (Warudo 로 보낼 값). [..., 87] 배치 가능
rig = rt.vrm_to_rig(vrm_frame87)         # 트래커 값 → 모델 입력 규약
rt.rig_tpose_frame()                     # VRM 단위 회전에 해당하는 리그 프레임
```

---

## 8. 문제 해결

| 증상 | 원인 / 조치 |
|---|---|
| `체크포인트 폴더를 찾을 수 없습니다` | `--run` 철자. 메시지에 사용 가능한 런 목록이 나온다 |
| `--large` 에서 load_state_dict 오류 | `--run` 을 AI_model_large 런 폴더로 주지 않았다 |
| 숫자가 예전 보고와 다르다 | 기본 에폭이 바뀌었다(§1). `--epoch` 로 고정해 비교 |
| 브리지가 `수신 대기…` 에서 멈춤 | 트래커 송신 포트가 `--listen-port` 와 다르거나, 21본 중 일부를 안 보낸다(메시지에 빠진 본 표시) |
| 아바타가 뒤틀림 | `--no-retarget` 을 켰거나, Warudo 모델이 정규화 VRM 이 아님(Enforce T-Pose 로 재출력) |
| 아바타가 떨림 | 트래커가 Warudo 로도 직접 보내고 있다 |
| 아바타가 바닥에 묻히거나 뜸 | 재생 시 `--replay-hips-height` / `--hips-height` 조정 |
| 팔 롤·손바닥·발 기울기만 이상 | 리그 T-포즈의 뼈축 롤 자유도. `derive_rest_pose.py` 의 규칙 조정(§6) |
| selftest T15 실패 (포트) | 39565/39566 을 다른 프로그램이 사용 중 |
| 첫 1초 동안 보정이 안 된다 | 정상 — 30프레임 버퍼가 찰 때까지 원본 통과(워밍업) |
| `K깎임` 이 많다 | 잔여 관통(`proj_residual_cm`)이 0 이면 무해. 남으면 `--no-adaptive-k` 와 비교 |
| `server.py` 가 `이미 실행 중인 supervisor` 로 끝남 | 다른 창에서 이미 돌고 있다. `python AI_Streaming/server.py --status` / `--stop` |
| 서버 로그에 `브리지 비정상 종료` 가 반복 | `logs/server.log` 에서 직전 `[bridge …]` 줄 확인. 흔한 원인: 포트 39540 을 다른 프로그램이 잡음, `epoch` 철자·없는 체크포인트 |

---

## 9. 상시 실행 — `server.py` (로컬 서버)

`vmc_bridge.py` 를 콘솔에서 켜 두는 대신 **supervisor 가 자식 프로세스로 띄우고 감시**한다.
브리지 코드는 그대로이고, supervisor 는 아래 일을 더한다 (2026-10-04 실측으로 필요성이 드러난 것들).

| 문제 (supervisor 없이) | server.py 가 하는 일 |
|---|---|
| 브리지가 죽으면 그대로 끝 | 비정상 종료 시 재시작, 백오프 1→2→4…60초 (5분 버티면 1초로 복귀) |
| 메인 루프가 멈춰도 프로세스는 살아 있어 모름 | 브리지 `/health` heartbeat 감시 → `stall_restart_s`(15초) 응답 없으면 강제 재시작 |
| 출력이 콘솔뿐 | `logs/server.log` 회전 로그 (5MB × 5개) — 브리지 출력도 `[bridge pid]` 로 함께 |
| 기본 에폭이 '폴더 최대'라 재시작 때 모델이 몰래 바뀜 | `server_config.json` 의 `epoch` 로 **고정** (현재 4000) |
| 두 번 켜면 포트 충돌 | `logs/server.lock` 배타 잠금 → 두 번째 실행은 코드 2 로 거부 |
| 조작은 콘솔 키(b/q)뿐 | 로컬 HTTP(127.0.0.1 전용) 로 상태 조회·bypass·리셋·종료 |

```bash
python AI_Streaming/server.py                 # 실행 (전경). Ctrl+C = 정상 종료
AI_Streaming\start_server.bat                 # 같은 것 (더블클릭용, UTF-8 콘솔 설정 포함)
python AI_Streaming/server.py --status        # 다른 창: 상태 JSON
python AI_Streaming/server.py --bypass on     # 원본 통과 (off / toggle)
python AI_Streaming/server.py --reset         # 세션 리셋 (정규화 기준·링버퍼)
python AI_Streaming/server.py --stop          # supervisor + 브리지 정상 종료
python AI_Streaming/server.py --print-cmd     # 띄울 브리지 명령만 출력 (설정 확인용)
python AI_Streaming/server.py --config other.json
```

### `server_config.json`

| 키 | 기본 | 설명 |
|---|---|---|
| `run` / `epoch` | l10.1_phys0.5 / 4000 | 로드할 체크포인트. `epoch: null` 이면 폴더 최대 (경고 로그) |
| `listen_port` / `send_host` / `send_port` / `fps` | 39540 / 127.0.0.1 / 39539 / 30 | vmc_bridge 와 같음 |
| `status_port` | 39580 | 상태·제어 HTTP (127.0.0.1 에만 열림, 인증 없음) |
| `threads` | 1 | torch CPU 스레드. 아래 실측 참고 |
| `log_csv` | null | 프레임별 CSV 경로. 상시 실행에서는 회전이 없어 **약 6.5MB/시간** 늘어난다 — 문제 조사 때만 |
| `emit_lag` | 2 | 브리지 `--emit-lag` (출력 지연 = emit_lag × 33ms) |
| `hold_repush` | false | true 면 브리지 `--hold-repush` (2026-10-07 이전 동작) |
| `extra_args` | [] | 브리지에 그대로 붙일 인자 (예: `["--no-adaptive-k"]`) |
| `log_dir` / `log_max_mb` / `log_backups` | logs / 5 / 5 | 회전 로그 |
| `health_interval_s` / `startup_grace_s` / `stall_restart_s` | 2 / 90 / 15 | 감시 주기 / 첫 정상 응답 전 유예 / 멈춤 판정 |
| `stable_after_s` / `max_backoff_s` | 300 / 60 | 백오프 복귀 기준 / 상한 |

### 상태 HTTP (브리지 `--status-port`)

| 요청 | 응답 |
|---|---|
| `GET /health` | `{"ok":true,"heartbeat_age_s":…}` — 메인 루프 heartbeat 가 5초 넘게 멈추면 **503** |
| `GET /status` | 가동 시간, 체크포인트, 입·출력 프레임, 트래커 무신호 초, idle, bypass, 최근 600프레임 p95/max ms, 예산 초과 %(최근 창·누적), 오류·송신 실패 수, torch 스레드 |
| `POST /bypass?on=1` / `?on=0` / (없음=토글) | 202 — 메인 루프가 다음 프레임 사이에 적용 |
| `POST /reset`, `POST /shutdown` | 202 |

`/shutdown` 으로 브리지가 코드 0 으로 끝나면 supervisor 는 '의도된 종료'로 보고 함께 끝난다(재시작 안 함).

### 로그온 시 자동 시작 (선택 — 직접 등록)

```powershell
schtasks /Create /TN "MCC_VMC_Bridge" /SC ONLOGON /RL LIMITED /TR "\"C:\Users\Contemplator\Desktop\MCC\AI_Streaming\start_server.bat\""
schtasks /Run /TN "MCC_VMC_Bridge"       # 지금 바로 실행
schtasks /Delete /TN "MCC_VMC_Bridge" /F  # 해제
```

Windows 서비스(로그온 없이 실행)로는 만들지 않았다 — Warudo 자체가 로그온 세션의 데스크톱 앱이라
브리지도 같은 세션에서 도는 것이 맞다. `start_server.bat` 는 Python 경로를 절대 경로로 적어 두었다
(작업 스케줄러는 사용자 PATH 를 다르게 볼 수 있다). Python 을 옮기면 이 줄을 고친다.

### 실측 (2026-10-04, CPU, persistent 손상 tracker_sim → UDP → 브리지, Warudo 없음)

| 항목 | 수정 전 브리지 단독 20분 | server.py, threads=1, 10분 |
|---|---|---|
| 출력 fps | 30.00 | 29.99 |
| 보정 예외 / 송신 실패 | 0 / 0 | 0 / 0 |
| 보정 프레임 잔여 관통 max | 0.000cm | 0.000cm |
| 처리 p50 / p95 / p99 / max | 8.1 / 9.8 / 18.1 / 36.7 ms (마지막 5분) | 13.6 / 20.1 / 27.5 / 78.4 ms |
| 예산(33.3ms) 초과 | 0.03% (마지막 5분) | 0.18% (33/18146) |
| **프로세스 CPU** | **5.11 코어 상시** | **0.42 코어** |

- **CPU 5코어**는 계산이 아니라 torch 의 기본 스레드(6)가 프레임 사이에 바쁜 대기(spin)하는 비용이다.
  30fps 페이싱 run_sim 600프레임에서 스레드 1/2/4/6 = CPU 0.26/1.17/2.92/4.90 코어, p95 16.6/13.8/15.5/13.1ms,
  **보정 결과(관통·이동량)는 전부 동일**. Warudo 와 같은 PC 에서 CPU 를 나눠 쓰므로 기본값을 1 로 두었다.
  server.py 10분 비교에서 `threads: 2` 는 p95 20.9ms·초과 0.19% 로 **1 과 같고 CPU 만 1.33코어**였다 — 2 로 올릴 이유 없음.
  수정 전(6스레드 spin) p95 9.8ms 와의 차이는 스레드 수가 아니라 spin 이 코어를 깨워 두는 효과로 보인다(미검증).
  그 지연이 꼭 필요하면 `threads: 0`(torch 기본 = 5코어 상시).
- 장시간 결함 (수정 완료): `corrector.summary()` 가 전 구간 리스트를 정렬해 **1시간 분량 29ms, 8시간 352ms,
  24시간 1.4초** 동안 브리지 루프를 멈췄다(5초마다 호출) + 통계 메모리 ≈14MB/시간. 이제 분위수는 최근 9000프레임
  (5분), 개수·합·최대·초과 수는 누적값 → 24시간 분량에서 1.8ms. 창보다 짧은 실행에서는 숫자가 예전과 같다.
- 송신 실패(`OSError`, 예: Warudo 가 다른 PC 이고 네트워크가 끊김)는 예전에 try 밖이라 **브리지 전체가 죽었다**.
  이제 세고(`send_errors`) 다음 프레임에서 다시 보낸다.
- 장애 주입 10/10: 기동 / 중복 실행 거부 / 원격 bypass / 강제 종료 → 3.5초 뒤 재시작·트래커 재수신 /
  프로세스 정지(멈춤) → 12.1초 뒤 재시작 / 트래커 끊김 → idle / `--stop` 정상 종료 / 로그 기록.
