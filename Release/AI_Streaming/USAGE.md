# AI_Streaming 사용법 상세

README 가 "무엇이고 어떻게 동작하나"라면 이 문서는 **"어떻게 쓰나"** 다.
모든 명령은 **Release 루트**(`AI_Streaming/` 의 부모 폴더 = 저장소 최상위)에서 실행하는 것을 기준으로 적었다.
(`paths.py` 가 절대 경로를 쓰므로 다른 곳에서 실행해도 같은 파일을 보지만, 상대 경로로 준
`--log-csv` 같은 출력 파일은 실행 위치에 생긴다.)

- 설치: `pip install -r requirements.txt` (`torch`, `numpy`, `pandas`, `python-osc`).
- 콘솔 한글이 깨지면 PowerShell 에서 `$env:PYTHONIOENCODING="utf-8"` 를 먼저 둔다(`start_server.bat` 는 자동).
- 장치는 **CPU 권장**. 한 번에 30프레임 윈도우 하나만 도는 구조라 CUDA 는 오히려 3~6배 느렸다(실측).
- 데이터셋이 필요한 검증·시연 도구는 이 폴더에 없다. `vmc_bridge.py` 의 `--replay`(`MODE="replay"`)도 데이터셋이
  없으면 동작하지 않는다. `--loopback-test`(`MODE="loopback"`)는 합성 모션을 쓰므로 **데이터셋 없이 돈다**(§0 단계 1).

---

## 0. 처음 쓸 때의 순서

| 단계 | 할 일 | 확인 |
|---|---|---|
| 0 자체 점검 | `python AI_Streaming/vmc_bridge.py --loopback-test` (Warudo·트래커 불필요, 약 10초) | `[loopback] 관통 제거: … 관통 max 0.000 cm` + `[loopback] OK`, 종료 코드 0 |
| 1 설정 확인 | `python AI_Streaming/server.py --print-cmd` | 브리지 명령줄에 `--emit-lag 2` 가 보임 |
| 2 브리지 실행 | `AI_Streaming\start_server.bat` (또는 `python AI_Streaming/server.py`) | 로그에 `수신 대기…` |
| 3 트래커 연결 | 트래커 앱의 VMC 송신 대상을 `127.0.0.1:39540` 으로 | `수신 대기…` 가 사라지고 콘솔 통계에 `in`·`트래커 Nf` 가 늘어남 |
| 4 Warudo 연결 | 캐릭터 → 모션 캡처 → VMC Receiver 포트 39539, VRM 모델 | 아바타가 뒤틀리지 않음, 팔 롤·손바닥·발 기울기·골반 높이 정상 |
| 5 상태 확인 | `python AI_Streaming/server.py --status` | `errors`·`send_errors` 0, `over_budget_pct_recent` ≈0, `hold` 가 계속 크게 늘지 않음 |

처음에는 보정 캐릭터 옆에 트래커 원본을 직접 받는 캐릭터를 하나 더 두고(트래커 앱이 두 곳으로 보낼 수 있으면
다른 포트, 예: 39541) 나란히 비교하면 확인이 쉽다. 보정 캐릭터가 원본보다 2프레임(67ms) 늦는 것은 정상이다.

---

## 1. 어떤 모델이 로드되나

| 실행 방법 | 로드되는 체크포인트 |
|---|---|
| `server.py` / `start_server.bat` | `checkpoints/temp.pth` |
| `vmc_bridge.py` 직접 실행 | `checkpoints/temp.pth` |

가중치는 **`checkpoints/temp.pth` 파일 하나**다(하위 폴더 없음, 이름은 `dataset_pipeline.CHECKPOINT_FILENAME`).
고를 것이 없으므로 파일을 바꿔 넣지 않는 한 재시작해도 같은 모델이 로드된다. `--run` / `--epoch` 옵션은 없다.
배포 가중치는 개발판 `tfm_declip_cov_l10.1_anat_recon1_phys0.5_kl0` 의 epoch 4000 이다(이 문서의 실측 숫자도 이 모델).
파일이 없으면 브리지가 `가중치 파일이 없습니다: …checkpoints\temp.pth` 로 멈춘다.

---

## 2. 상시 실행 — `server.py` (권장)

`vmc_bridge.py` 를 콘솔에서 켜 두는 대신 **supervisor 가 자식 프로세스로 띄우고 감시**한다.

| 문제 (supervisor 없이) | server.py 가 하는 일 |
|---|---|
| 브리지가 죽으면 그대로 끝 | 비정상 종료 시 재시작, 백오프 1→2→4…60초 (5분 버티면 1초로 복귀) |
| 메인 루프가 멈춰도 프로세스는 살아 있어 모름 | 브리지 `/health` heartbeat 감시 → `stall_restart_s`(15초) 응답 없으면 강제 재시작 |
| 출력이 콘솔뿐 | `logs/server.log` 회전 로그 (5MB × 5개) — 브리지 출력도 `[bridge pid]` 로 함께 |
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
| `listen_port` / `send_host` / `send_port` / `fps` | 39540 / 127.0.0.1 / 39539 / 30 | §3 의 브리지 옵션과 같음. Warudo 가 다른 PC 면 `send_host` 를 그 PC 주소로 |
| `status_port` | 39580 | 상태·제어 HTTP (127.0.0.1 에만 열림, 인증 없음) |
| `threads` | 1 | torch CPU 스레드. 아래 실측 참고 |
| `log_csv` | null | 프레임별 CSV 경로. 상시 실행에서는 회전이 없어 **약 6.5MB/시간** 늘어난다 — 문제 조사 때만 |
| `emit_lag` | 2 | 브리지 `--emit-lag` (출력 지연 = emit_lag × 33ms). 0 = 지연 최소, 튐 큼 |
| `hold_repush` | false | true 면 브리지 `--hold-repush` (2026-10-07 이전 동작) |
| `extra_args` | [] | 브리지에 그대로 붙일 인자 (예: `["--no-adaptive-k"]`) |
| `log_dir` / `log_max_mb` / `log_backups` | logs / 5 / 5 | 회전 로그 |
| `health_interval_s` / `startup_grace_s` / `stall_restart_s` | 2 / 90 / 15 | 감시 주기 / 첫 정상 응답 전 유예 / 멈춤 판정 |
| `stable_after_s` / `max_backoff_s` | 300 / 60 | 백오프 복귀 기준 / 상한 |

### 상태 HTTP (브리지 `--status-port`)

| 요청 | 응답 |
|---|---|
| `GET /health` | `{"ok":true,"heartbeat_age_s":…}` — 메인 루프 heartbeat 가 5초 넘게 멈추면 **503** |
| `GET /status` | 가동 시간, 체크포인트, 입·출력 프레임, 트래커 무신호 초, idle, bypass, hold, emit_lag, 최근 600프레임 p95/max ms, 예산 초과 %(최근 창·누적), 오류·송신 실패 수, torch 스레드 |
| `POST /bypass?on=1` / `?on=0` / (없음=토글) | 202 — 메인 루프가 다음 프레임 사이에 적용 |
| `POST /reset`, `POST /shutdown` | 202 |

`/shutdown` 으로 브리지가 코드 0 으로 끝나면 supervisor 는 '의도된 종료'로 보고 함께 끝난다(재시작 안 함).

### 로그온 시 자동 시작 (선택 — 직접 등록)

```powershell
schtasks /Create /TN "MCC_VMC_Bridge" /SC ONLOGON /RL LIMITED /TR "\"<Release 루트>\AI_Streaming\start_server.bat\""
schtasks /Run /TN "MCC_VMC_Bridge"       # 지금 바로 실행
schtasks /Delete /TN "MCC_VMC_Bridge" /F  # 해제
```

Windows 서비스(로그온 없이 실행)로는 만들지 않았다 — Warudo 자체가 로그온 세션의 데스크톱 앱이라
브리지도 같은 세션에서 도는 것이 맞다. `start_server.bat` 는 PATH 의 `python` 을 쓴다. 작업 스케줄러는 사용자
PATH 를 다르게 볼 수 있으므로, 자동 시작이 안 되면 배치 파일의 `python` 을 `python.exe` 의 전체 경로로 바꾼다.

### 실측 (2026-10-04, CPU, 손상 입력 가상 트래커 → UDP → 브리지, Warudo 없음)

| 항목 | 브리지 단독(torch 기본 스레드) 20분 | server.py, threads=1, 10분 |
|---|---|---|
| 출력 fps | 30.00 | 29.99 |
| 보정 예외 / 송신 실패 | 0 / 0 | 0 / 0 |
| 보정 프레임 잔여 관통 max | 0.000cm | 0.000cm |
| 처리 p50 / p95 / p99 / max | 8.1 / 9.8 / 18.1 / 36.7 ms (마지막 5분) | 13.6 / 20.1 / 27.5 / 78.4 ms |
| 예산(33.3ms) 초과 | 0.03% (마지막 5분) | 0.18% (33/18146) |
| **프로세스 CPU** | **5.11 코어 상시** | **0.42 코어** |

- **CPU 5코어**는 계산이 아니라 torch 의 기본 스레드가 프레임 사이에 바쁜 대기(spin)하는 비용이다.
  보정 결과(관통·이동량)는 스레드 수와 무관하게 동일하다. Warudo 와 같은 PC 에서 CPU 를 나눠 쓰므로 기본값을 1 로 두었다.
  `threads: 2` 는 p95·초과율이 1 과 같고 CPU 만 1.33코어였다. 지연이 꼭 더 낮아야 하면 `threads: 0`(torch 기본 = 5코어 상시).
- 장애 주입 10/10: 기동 / 중복 실행 거부 / 원격 bypass / 강제 종료 → 3.5초 뒤 재시작·트래커 재수신 /
  프로세스 정지(멈춤) → 12.1초 뒤 재시작 / 트래커 끊김 → idle / `--stop` 정상 종료 / 로그 기록.

---

## 3. `vmc_bridge.py` — 실제 송수신 (트래커 → 보정 → Warudo)

```
트래커 ─VMC/UDP─▶ [--listen-port] 수신 → VRM→리그 변환 + 세션 정규화 → 보정 → 되돌림 → [--send-port] ─▶ Warudo
```

`server.py` 가 이 파일을 띄운다. 콘솔에서 직접 돌리는 것은 설정을 시험하거나 문제를 볼 때다.

### 실행 형태

**VSCode "Run Python File"**: `vmc_bridge.py` 위쪽 `[실행 설정]` 블록의 포트·옵션 상수를 바꾸고 인자 없이
실행한다(방송은 `MODE="live"`, 자체 점검은 `"loopback"`). 실행하면 같은 설정의 명령줄이 `[실행 설정] python ...` 로 출력된다.
인자를 하나라도 주면 그 블록은 무시된다(server.py 영향 없음).

```bash
python AI_Streaming/vmc_bridge.py --loopback-test                       # 자체 점검 (Warudo·데이터셋 불필요)
python AI_Streaming/vmc_bridge.py --listen-port 39540 --send-port 39539 # 라이브
python AI_Streaming/vmc_bridge.py --listen-port 39540 --dry-run         # 수신·보정만, 전송 안 함
python AI_Streaming/vmc_bridge.py --listen-port 39540 --log-csv bridge_log.csv
python AI_Streaming/vmc_bridge.py --listen-port 39540 --no-retarget     # [시연] 변환 없이 → 뒤틀림
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--listen-port` | 39540 | 트래커가 보내는 포트. **트래커 앱의 송신 대상을 이 포트로** 바꾼다 |
| `--send-host` / `--send-port` | 127.0.0.1 / 39539 | Warudo VMC Receiver 주소 (Warudo 기본 39539) |
| `--fps` | 30 | 브리지 시계. 트래커가 더 빠르면 최신 프레임만 쓰고, 새 프레임이 없는 틱(hold)은 모델에 넣지도 보내지도 않는다(보간 없음) |
| `--emit-lag` | **2** | 윈도우 끝에서 몇 프레임 앞을 내보낼지. 2 = +67ms 지연으로 끝자리 튐 완화, 0 = 지연 최소(튐 큼) |
| `--hold-repush` | 끔 | hold 틱에도 같은 프레임을 모델에 다시 넣는다 — 2026-10-07 이전 동작(비교·롤백용) |
| `--threads` | 0 (torch 기본) | torch CPU 스레드. server.py 는 1 로 띄운다(§2 실측) |
| `--dry-run` | 끔 | 송신 안 함 |
| `--no-fold-upperchest` | 끔 | 트래커의 UpperChest 를 Chest 에 접지 않는다 |
| `--no-lowpass` / `--no-projection` | 끔 | 단계를 빼고 대조 (사영을 빼면 관통이 남는다) |
| `--no-adaptive-k` | 끔 | 예산 가드를 끄고 사영 반복 K=24 고정 |
| `--no-retarget` | 끔 | [디버그] 리그↔VRM 변환을 끈다. 아바타가 뒤틀리는 것이 정상 |
| `--stats-every` | 5.0 | 콘솔 통계 주기(초) |
| `--idle-after` | 0.5 | 트래커 수신이 이만큼 없으면 송신 정지 (Warudo 는 마지막 포즈 유지) |
| `--reset-after` | 2.0 | 이만큼 이상 끊겼다 돌아오면 세션 리셋(정규화 기준·링버퍼) → 재개 첫 1초는 원본 통과 |
| `--log-csv` | 없음 | 프레임별 기록 CSV |
| `--loopback-test` | 끔 | 합성 트래커(yaw 37°) → 브리지 → 로컬 수신기. **데이터셋 불필요** — 매 프레임 4쌍이 관통하는 합성 모션(2.0~5.2cm)을 보내 UDP·번들 파싱·Hips 절대 위치 복원·정규화·pass-through·**수신된 보정 프레임의 관통 0**·보정 예외 0 을 검사 (포트 39555/39556) |
| `--replay` | — | **데이터셋이 필요해 Release 에서는 동작하지 않는다** |

실행 중 키: **`b`** = bypass 토글(원본 통과, 패닉 스위치), **`q`** = 종료. (Windows 콘솔에서만 — `msvcrt`.
server.py 로 띄운 브리지는 키를 읽지 않는다 → `server.py --bypass` / `--stop`)

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

30fps 트래커에서 hold 는 보통 전체 틱의 몇 % 이하다(실측 0.1~3%). 계속 크게 늘면 트래커 fps 설정이나 CPU 여유를 본다.

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

## 4. Python 에서 직접 쓰기 (API)

모든 모듈은 `AI_Streaming` 을 `sys.path` 에 넣고 import 한다. `paths` 를 먼저 import 하면 `AI_model` 도 경로에 올라간다.

```python
import sys; sys.path.insert(0, r"<Release 루트>\AI_Streaming")
import paths, contract, sinks, retarget
from corrector import StreamingCorrector
```

### 4.1 보정기 — `StreamingCorrector`

```python
ckpt = paths.checkpoint_path()                    # checkpoints/temp.pth 의 절대 경로
corr = StreamingCorrector(ckpt, emit_lag=2)       # 생성 시 모델 로드 + 1회 워밍업(약 0.15초)

for frame in my_frames:                           # frame: [87] 텐서 (리그 규약, §4.3)
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
| `stride`, `emit_lag` | 1, 0 | `stride+emit_lag ≤ 30`. 지연 `corr.delay = stride-1+emit_lag`. **브리지는 emit_lag 2 로 만든다** |
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

### 4.2 출력부 — `sinks`

```python
rec = sinks.RecordSink();  rec.send(out);  rec.tensor()        # [N, 87] 모아두기
osc = sinks.VmcOscSink()                                       # DRY-RUN (보내지 않음)
osc = sinks.VmcOscSink(client=SimpleUDPClient("127.0.0.1", 39539), hips_offset=(0, 0.9, 0))
osc.send(out)                   # 리그 규약 프레임 → VRM 변환 → OSC 번들 1개
osc.send(vrm_frame, already_vrm=True)   # 이미 VRM 규약이면 변환 건너뜀
print(osc.report())
```

### 4.3 프레임 규약과 검사 — `contract`

87차원 = `[0:3]` Hips 위치(m) + `[3:87]` 21관절 로컬 쿼터니언 `(x,y,z,w)`, 관절 순서는 알파벳순
(`contract.BONE_ORDER`). 30fps, 윈도우 30프레임.

```python
f = contract.pack_frame(hips_xyz_m, {"Hips": (x,y,z,w), "Chest": ..., ...})   # 21본 모두 필요 (빠지면 KeyError)
hips, quats = contract.unpack_frame(f)
problems = contract.validate_frame(f, physics=contract.make_physics())       # [] 이면 정상
```

### 4.4 리타깃 — `retarget.Retarget`

```python
rt = retarget.Retarget()                 # rig_tpose.json 로드
vrm = rt.rig_to_vrm(frame87)             # 리그 로컬 → VRM 로컬 (Warudo 로 보낼 값). [..., 87] 배치 가능
rig = rt.vrm_to_rig(vrm_frame87)         # 트래커 값 → 모델 입력 규약
rt.rig_tpose_frame()                     # VRM 단위 회전에 해당하는 리그 프레임
```

---

## 5. 문제 해결

| 증상 | 원인 / 조치 |
|---|---|
| `가중치 파일이 없습니다` | `checkpoints` 폴더 바로 안에 `temp.pth` 가 있어야 한다 (하위 폴더에 넣으면 못 찾는다) |
| `No module named 'sources'` | `MODE="replay"` / `--replay` 로 실행했다(데이터셋 필요). 방송은 `MODE="live"` |
| `--loopback-test` 가 `보정 프레임에 관통이 남았습니다` 로 실패 | `--no-projection` 을 줬거나 CPU 가 밀려 예산 가드가 사영을 깎았다(메시지의 K 깎임 횟수). 다른 무거운 프로그램을 끄고 다시 |
| `--loopback-test` 가 `보정 예외 N건` 으로 실패 | 보정 중 예외 — 방송이라면 원본이 그대로 나갔을 상황. 설치(torch 버전)·체크포인트 파일 손상을 의심 |
| `--loopback-test` 가 포트 오류 | 39555/39556 을 다른 프로그램이 사용 중 |
| 브리지가 `수신 대기…` 에서 멈춤 | 트래커 송신 포트가 `--listen-port` 와 다르거나, 21본 중 일부를 안 보낸다(메시지에 빠진 본 표시) |
| 아바타가 뒤틀림 | `--no-retarget` 을 켰거나, Warudo 모델이 정규화 VRM 이 아님(Enforce T-Pose 로 재출력) |
| 아바타가 떨림 | 트래커가 Warudo 로도 직접 보내고 있다 |
| 보정 아바타가 가끔 툭 튐 | 알려진 한계(README '보정 아바타 튐'). `emit_lag` 가 0 으로 바뀌지 않았는지 `--print-cmd` 로 확인 |
| 첫 1초 동안 보정이 안 된다 / 1초 뒤 한 번 툭 움직인다 | 정상 — 30프레임 버퍼가 찰 때까지 원본 통과(워밍업), 그 뒤 보정으로 넘어가는 순간의 전환 |
| 팔 롤·손바닥·발 기울기만 이상 | 리그 T-포즈의 뼈축 롤 자유도(README 'Warudo 호환 1'). 아바타별로 다를 수 있다 |
| `K깎임` 이 많다 | 잔여 관통(`proj_residual_cm`)이 0 이면 무해. 남으면 `--no-adaptive-k` 와 비교 |
| `hold` 가 계속 크게 늘어난다 | 트래커가 30fps 보다 느리게 보내거나 CPU 가 밀린다. 트래커 fps 설정, `server.py --status` 의 p95 확인 |
| `server.py` 가 `이미 실행 중인 supervisor` 로 끝남 | 다른 창에서 이미 돌고 있다. `python AI_Streaming/server.py --status` / `--stop` |
| 서버 로그에 `브리지 비정상 종료` 가 반복 | `logs/server.log` 에서 직전 `[bridge …]` 줄 확인. 흔한 원인: 포트 39540 을 다른 프로그램이 잡음, `checkpoints/temp.pth` 없음 |
