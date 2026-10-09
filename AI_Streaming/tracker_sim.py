"""가상 모션캡처 장비 — held-out 데이터셋을 '트래커가 실시간으로 보내는 것처럼' VMC(UDP)로 송신한다.

    VSCode "Run Python File" — 아래 [실행 설정] 블록을 바꾸고 인자 없이 실행 (기본 = Warudo 3-아바타 비교)
    python AI_Streaming/tracker_sim.py                                  # clean, 30fps, 127.0.0.1:39540 으로
    python AI_Streaming/tracker_sim.py --scenario persistent --to 127.0.0.1:39540 --to 127.0.0.1:39541
    python AI_Streaming/tracker_sim.py --scenario legacy80 --clean-to 127.0.0.1:39542 --loop
    python AI_Streaming/tracker_sim.py --fps 60                         # 60fps 트래커 흉내 (브리지 리샘플링 검사)
    python AI_Streaming/tracker_sim.py --no-retarget --to 127.0.0.1:39539   # [시연] 변환 없이 보내면 뒤틀리는 것 확인

[왜 별도 프로세스인가]
  vmc_bridge --replay 는 같은 프로세스 안에서 재생하므로 UDP 수신·번들 파싱·타이밍 경로를 타지 않는다.
  이 스크립트는 **실제 장비와 같은 자리**(다른 프로세스, UDP, 자기 시계)에서 보내므로 브리지·Warudo 는
  진짜 트래커와 구분할 수 없다. 같은 프레임을 여러 대상으로 보낼 수 있어 Warudo 에서 캐릭터를 나란히
  놓고 비교할 수 있다:
      손상 입력  : --to <브리지 포트>  +  --to <Warudo 캐릭터 A 포트>   (같은 손상 프레임)
      보정 출력  : 브리지가 Warudo 캐릭터 B 포트로 보낸다
      클린 참조  : --clean-to <Warudo 캐릭터 C 포트>  (손상 주입 전 원본 — 정답)

[무엇을 보내는가]
  sources.ReplaySource 가 30프레임 청크마다 시나리오별 손상을 주입한 리그 프레임을 내놓는다.
  이를 retarget.rig_to_vrm 으로 VRM 규약으로 바꾸고 Hips 에 --hips-height 를 더해(데이터셋 Hips 는
  첫 프레임 상대값 ≈ 0 이라 그대로면 아바타가 바닥에 파묻힌다) 프레임당 번들 1개로 보낸다.
  페이싱은 ReplaySource(pace=True)의 단조 시계 — 밀리면 따라잡지 않고 건너뛴다(실제 장비와 같다).

[출력 통계 5초마다]
  보낸 프레임 수, 실측 fps, 주입 관통(4쌍 max, cm) — 이 값이 '입력에 무엇이 들어 있었나'의 기록이다.
  **보정 전** 값이다: 브리지로 보내기 전의 손상 프레임을 잰 것이라 브리지가 무엇을 하든 줄지 않는다.
  보정 결과(출력 관통)는 브리지 콘솔의 `관통 4쌍 max 입력 … → 출력 …` 줄에서 본다.
  최근 = 이번 통계 창(5초)의 max, 누적 = 시작부터의 max. (2026-10-09 전에는 누적만 찍어 한 번 깊은
  주입이 지나가면 숫자가 고정됐다 — persistent 기본 설정에서 42초째 8.62cm 이후 78초째까지 그대로.)
"""

import argparse
import csv
import os
import sys
import time

import paths

import torch

import contract
import retarget as rt
import sinks
import sources

# =====================================================================
# [실행 설정] VSCode "Run Python File" 용 — train.py·vmc_bridge.py 와 같은 규칙: 여기 값을 바꾸고 실행한다.
#   인자 없이 실행했을 때(= Run Python File, F5)만 이 값들을 쓴다.
#   명령줄 인자를 하나라도 주면 이 블록은 무시되고 argparse 기본값이 쓰인다(README·USAGE 의 명령어 동작 불변).
#   실행 시 이 설정과 같은 명령줄을 출력하므로 터미널에서 재현할 때 그대로 복사하면 된다.
#
#   기본값 = Warudo 3-아바타 비교 (vmc_bridge 는 MODE="live" 로 먼저 띄운다):
#     손상 입력 → 브리지(39540) + 캐릭터 A(39541)  |  브리지 보정 → 캐릭터 B(39539)  |  원본 → 캐릭터 C(39542)
# =====================================================================
SCENARIO = "persistent"     # "clean" | "transient" | "persistent" | "legacy80"
TO = ["127.0.0.1:39540",    # 손상 입력 대상 (반복 가능): 브리지 입력 포트
      "127.0.0.1:39541"]    #                            캐릭터 A (손상 입력 그대로)
CLEAN_TO = ["127.0.0.1:39542"]   # 원본(정답) 대상: 캐릭터 C. 빈 목록 [] = 보내지 않음
LOOP = True                 # True = 파일 끝에서 처음부터 반복 (q 로 종료)
SECONDS = 0                 # 송신 시간(초). 0 = 파일 끝까지 / LOOP 면 무한
FILES = 8                   # held-out 파일 수
FPS = contract.FPS          # 송신 프레임률. 60 이면 브리지의 sample-and-hold 검사
HIPS_HEIGHT = 0.9           # Hips 절대 높이(m). 아바타가 뜨거나 잠기면 조정
LEVEL_GROUND = True         # 데이터셋의 경사 바닥을 평지로 편다. False = 원본 그대로(비스듬히 올라감)
SEED = 20260707             # 손상 주입 시드 (같은 시드 = 같은 손상)
LOG_CSV = None              # 예: "tracker_log.csv" — 프레임별 송신 기록 (실행 위치 기준 상대 경로)


def _argv_from_settings():
    """위 [실행 설정] 블록 → 명령줄 인자 목록. 검증은 argparse 가 그대로 한다."""
    argv = ["--scenario", SCENARIO]
    for t in TO:
        argv += ["--to", t]
    for t in CLEAN_TO:
        argv += ["--clean-to", t]
    if LOOP:
        argv.append("--loop")
    if SECONDS:
        argv += ["--seconds", str(SECONDS)]
    argv += ["--files", str(FILES), "--fps", str(FPS), "--hips-height", str(HIPS_HEIGHT), "--seed", str(SEED)]
    if not LEVEL_GROUND:
        argv.append("--no-level")
    if LOG_CSV:
        argv += ["--log-csv", LOG_CSV]
    return argv


def _parse_target(s):
    host, _, port = s.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError(f"host:port 형식이어야 합니다: {s}")
    return host, int(port)


def _kbhit_q():
    try:
        import msvcrt
        return msvcrt.kbhit() and msvcrt.getwch().lower() == "q"
    except ImportError:
        return False


def main(argv=None):
    if argv is None and len(sys.argv) == 1:          # Run Python File (인자 없음) → [실행 설정] 블록
        argv = _argv_from_settings()
        print("[실행 설정] python AI_Streaming/tracker_sim.py " + " ".join(argv))
    ap = argparse.ArgumentParser(description="가상 모션캡처 장비: 데이터셋 → VMC(UDP) 실시간 송신")
    ap.add_argument("--scenario", default="clean", choices=["clean", "transient", "persistent", "legacy80"])
    ap.add_argument("--to", type=_parse_target, action="append", default=None,
                    help="송신 대상 host:port (반복 가능). 기본 127.0.0.1:39540 (브리지 입력)")
    ap.add_argument("--clean-to", type=_parse_target, action="append", default=[],
                    help="손상 주입 전 원본(정답)을 보낼 대상 host:port (반복 가능)")
    ap.add_argument("--fps", type=float, default=contract.FPS, help="송신 프레임률 (트래커 흉내)")
    ap.add_argument("--files", type=int, default=8, help="held-out 파일 수")
    ap.add_argument("--seconds", type=float, default=0, help="송신 시간 (0 = 파일 끝까지 / --loop 면 무한)")
    ap.add_argument("--loop", action="store_true", help="파일 끝에서 처음부터 반복")
    ap.add_argument("--hips-height", type=float, default=0.9, help="Hips 절대 높이(m) — 아바타에 맞게 조정")
    ap.add_argument("--no-level", action="store_true",
                    help="데이터셋의 경사 바닥을 펴지 않는다 (원본 그대로 — 걷기·달리기에서 아바타가 비스듬히 올라간다)")
    ap.add_argument("--seed", type=int, default=20260707, help="손상 주입 난수 시드")
    ap.add_argument("--no-retarget", action="store_true",
                    help="[시연] 리그 값을 변환 없이 보낸다 — Warudo 에서 아바타가 뒤틀리는 것이 정상")
    ap.add_argument("--log-csv", default=None, help="프레임별 송신 기록 CSV (t_s, frame, pen_in_cm, pen_clean_cm)")
    ap.add_argument("--stats-every", type=float, default=5.0)
    args = ap.parse_args(argv)
    targets = args.to or [("127.0.0.1", 39540)]

    from pythonosc.udp_client import SimpleUDPClient
    physics = contract.make_physics("cpu")
    retg = False if args.no_retarget else rt.Retarget()
    offset = (0.0, args.hips_height, 0.0)
    outs = [sinks.VmcOscSink(client=SimpleUDPClient(h, p), retarget=retg, hips_offset=offset)
            for h, p in targets]
    cleans = [sinks.VmcOscSink(client=SimpleUDPClient(h, p), retarget=retg, hips_offset=offset)
              for h, p in args.clean_to]

    files = paths.test_motion_files(limit=args.files)
    src = sources.ReplaySource(files, scenario=args.scenario, fps=args.fps, pace=True,
                               seed=args.seed, physics=physics, loop=args.loop,
                               level=not args.no_level)

    def pen4(f):
        d = physics.get_penetration_depths_from_quats(
            f[:3].reshape(1, 3), f[3:].reshape(1, 84), contract.COLLIDING_PAIRS) * 100.0
        return float(d.max())

    print("=" * 74)
    print("가상 모션캡처 장비 (tracker_sim)")
    print(f"  시나리오   : {args.scenario} — {sources.describe_scenarios()[args.scenario]}")
    print(f"  송신 대상  : {', '.join(f'{h}:{p}' for h, p in targets)}"
          + (f"   | 클린 참조 → {', '.join(f'{h}:{p}' for h, p in args.clean_to)}" if args.clean_to else ""))
    print(f"  프레임률   : {args.fps:.0f} fps   Hips 높이 {args.hips_height} m   "
          f"{'변환 OFF (뒤틀림 시연)' if args.no_retarget else '리그→VRM 변환 ON'}")
    print(f"  입력       : held-out {len(files)}파일{' (반복)' if args.loop else ''}"
          + (f", {args.seconds:.0f}초" if args.seconds else ""))
    print("  [q] 종료")
    print("=" * 74)

    log = None
    if args.log_csv:
        log_f = open(args.log_csv, "w", newline="", encoding="utf-8")
        log = csv.writer(log_f)
        log.writerow(["t_s", "frame", "pen_in_cm", "pen_clean_cm"])

    t_start = time.perf_counter()
    t_stats = t_start
    n = n_stats = 0
    pen_max_in = pen_max_clean = 0.0              # 누적 (시작부터)
    win_in = win_clean = 0.0                      # 이번 통계 창
    frames_colliding = 0
    try:
        for inp, clean in src:
            if _kbhit_q():
                break
            for s in outs:
                s.send(inp)
            for s in cleans:
                s.send(clean)
            n += 1
            pi, pc = pen4(inp), pen4(clean)
            pen_max_in, pen_max_clean = max(pen_max_in, pi), max(pen_max_clean, pc)
            win_in, win_clean = max(win_in, pi), max(win_clean, pc)
            frames_colliding += int(pi > 0)
            if log:
                log.writerow([f"{time.perf_counter() - t_start:.4f}", n, f"{pi:.3f}", f"{pc:.3f}"])
                log_f.flush()
            now = time.perf_counter()
            if now - t_stats >= args.stats_every:
                fps = (n - n_stats) / (now - t_stats)
                print(f"  보냄 {n:6d}f  실측 {fps:5.1f} fps  | 입력(보정 전) 4쌍 관통 max 최근 {win_in:5.2f} / "
                      f"누적 {pen_max_in:5.2f} cm (충돌 프레임 {100.0 * frames_colliding / n:5.1f}%)  "
                      f"클린 최근 {win_clean:4.2f} / 누적 {pen_max_clean:4.2f} cm")
                t_stats, n_stats = now, n
                win_in = win_clean = 0.0
            if args.seconds and now - t_start >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if log:
            log_f.close()
    el = time.perf_counter() - t_start
    print(f"종료: {n}프레임 / {el:.1f}s = {n / max(el, 1e-9):.1f} fps  "
          f"| 입력(보정 전) 관통 누적 max {pen_max_in:.2f} cm, 충돌 프레임 {100.0 * frames_colliding / max(n, 1):.1f}%")
    for s in outs + cleans:
        print("  " + s.report().split("\n")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
