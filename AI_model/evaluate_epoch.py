"""에폭별 성능 평가 — 한 run 의 체크포인트들을 일정 간격으로 평가해 `evaluate_epoch.csv` 에 쌓는다.

[왜 별도 파일인가]
  `evaluate.py` 는 폴더에서 **가장 큰 에폭 하나**만 평가하고, 결과를 `evaluate_results.csv` 에
  추가한다. 학습을 500 에폭까지 늘렸을 때 필요한 것은 "마지막이 최선인가"이고, 그러려면
  에폭 축을 따라 훑어야 한다. 21개 지점 × 4시나리오 = 84행을 실험 로그 CSV 에 쏟아부으면
  기존 272행의 가독성이 무너지므로, **별도 파일에 쓴다.**

[지표는 복사하지 않는다]
  cOKS·3DPCK·intent_dyn·관통·지터 정의는 전부 `evaluate.py` 에 있다. 이 스크립트는 지표를
  다시 구현하지 않고 `evaluate.evaluate()` 를 그대로 호출한 뒤, 결과 행을 가로채 저장한다.
  ⇒ 여기서 나온 숫자는 `evaluate_results.csv` 의 같은 조건 행과 **정의상 동일**하다.

  가로채는 두 지점 (원본 파일은 수정하지 않는다):
    evaluate.find_latest_checkpoint_in -> 지정한 에폭의 .pth 를 돌려주도록 교체
    evaluate.append_results_csv        -> 행을 가로채 evaluate_epoch.csv 로 보냄
                                          (= 실험 로그 CSV 오염 방지)

[출력 형식]
  `evaluate_results.csv` 와 같은 열 + 맨 앞에 **`epoch`** 열 하나. 이어서 실행하면
  이미 기록된 (epoch, mode) 조합은 건너뛰므로, 학습이 진행되는 도중에 여러 번 돌려도 된다.

실행:
    python AI_model/evaluate_epoch.py                       # 기본 run, 20 간격, 있는 것 전부
    python AI_model/evaluate_epoch.py --step 20 --to 500
    python AI_model/evaluate_epoch.py --run <폴더명> --scenarios clean,persistent
    python AI_model/evaluate_epoch.py --limit 20            # 빠른 스모크 (파일 20개만)
    python AI_model/evaluate_epoch.py --large               # AI_model_large 아키텍처로
"""

import argparse
import csv
import glob
import io
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_OUT = "evaluate_epoch.csv"


def _existing_epochs(run_dir):
    """폴더에 있는 체크포인트 에폭 번호 목록 (정렬)."""
    out = []
    for p in glob.glob(os.path.join(run_dir, "pvtvae_epoch_*.pth")):
        m = re.search(r"pvtvae_epoch_(\d+)\.pth$", os.path.basename(p))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _already_done(out_path):
    """CSV 에 이미 있는 (epoch, mode) 집합. 중간에 끊겨도 이어서 돌리기 위한 것."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with io.open(out_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add((int(row["epoch"]), row["mode"]))
            except (KeyError, ValueError, TypeError):
                continue
    return done


def _append_rows(out_path, rows):
    """행들을 CSV 에 추가한다. 헤더는 첫 행의 키 순서를 따른다 (evaluate.py 의 열 순서 보존).

    [주의] 열이 늘어난 경우(evaluate.py 가 지표를 추가한 경우) 기존 파일의 헤더를 유지하고
    새 열은 버리는 대신, **헤더를 새로 쓰고 기존 행을 마이그레이션**한다 — evaluate.py 의
    append_results_csv 와 같은 정책이다. 과거 행의 새 열은 빈칸으로 남는다.
    """
    if not rows:
        return
    new_fields = list(rows[0].keys())
    old_rows, old_fields = [], []
    if os.path.exists(out_path):
        with io.open(out_path, encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            old_fields = list(r.fieldnames or [])
            old_rows = list(r)

    fields = old_fields + [k for k in new_fields if k not in old_fields] if old_fields else new_fields
    with io.open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in old_rows + [{k: v for k, v in r.items()} for r in rows]:
            w.writerow({k: row.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser(description="에폭별 성능 평가 -> evaluate_epoch.csv")
    ap.add_argument("--run", default=None,
                    help="체크포인트 run 폴더 이름. 생략하면 train.py 의 현재 설정으로 결정")
    ap.add_argument("--step", type=int, default=20, help="에폭 간격 (기본 20)")
    ap.add_argument("--from", dest="start", type=int, default=0, help="시작 에폭 (0=처음부터)")
    ap.add_argument("--to", dest="end", type=int, default=0, help="끝 에폭 (0=있는 것 전부)")
    ap.add_argument("--scenarios", default="", help="쉼표 구분. 생략 시 전체 4개")
    ap.add_argument("--limit", type=int, default=0, help="테스트 파일 수 제한 (0=전체 306)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="출력 CSV 경로")
    ap.add_argument("--redo", action="store_true", help="이미 기록된 지점도 다시 평가")
    ap.add_argument("--no-lowpass", action="store_true")
    ap.add_argument("--no-projection", action="store_true")
    ap.add_argument("--large", action="store_true",
                    help="AI_model_large 의 확장 아키텍처로 로드")
    args = ap.parse_args()

    os.chdir(_ROOT)          # evaluate.py 의 상대 경로 가정(processed_motions_VMC, checkpoints)

    import evaluate as ev
    import lowpass
    import projection
    from dataset_pipeline import make_run_name, resolve_ckpt_root

    if args.large:
        sys.path.insert(0, os.path.join(_ROOT, "AI_model_large"))
        from models_large import TransformerDenoiserLargeCompat
        import train_large as cfg
        ev.MODEL_CLASS = TransformerDenoiserLargeCompat
        ev.RUN_TAG, ev.LAMBDA_RECON = cfg.RUN_TAG, cfg.LAMBDA_RECON
        ev.LAMBDA_PHYS, ev.BETA_KL = cfg.LAMBDA_PHYS, cfg.BETA_KL

    run_name = args.run or make_run_name(ev.LAMBDA_RECON, ev.LAMBDA_PHYS, ev.BETA_KL, tag=ev.RUN_TAG)
    run_dir = os.path.join(resolve_ckpt_root(), run_name)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"❌ run 폴더가 없습니다: {run_dir}")

    if args.no_lowpass:
        lowpass.LP_ENABLED = False
    if args.no_projection:
        projection.PROJ_ENABLED = False

    epochs = _existing_epochs(run_dir)
    if not epochs:
        raise SystemExit(f"❌ 체크포인트가 없습니다: {run_dir}")
    targets = [e for e in epochs if e % args.step == 0]
    if args.start:
        targets = [e for e in targets if e >= args.start]
    if args.end:
        targets = [e for e in targets if e <= args.end]
    if not targets:
        raise SystemExit(f"❌ 조건에 맞는 에폭이 없습니다. 폴더에 있는 에폭: {epochs}")

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()] or ev.RUN_SCENARIOS
    out_path = os.path.join(_ROOT, args.out) if not os.path.isabs(args.out) else args.out
    done = set() if args.redo else _already_done(out_path)

    print("=" * 74)
    print("에폭별 평가")
    print(f"  run        : {run_name}")
    print(f"  에폭       : {targets}  (간격 {args.step})")
    print(f"  시나리오   : {scenarios}")
    print(f"  구성       : 저역통과 {'OFF' if args.no_lowpass else lowpass.LP_MODE} / "
          f"사영 {'OFF' if args.no_projection else 'K=%d' % projection.PROJ_K}")
    print(f"  출력       : {out_path}  (이미 기록된 지점 {len(done)}개는 건너뜀)")
    print("=" * 74)

    # --- evaluate.py 의 두 지점을 가로챈다 (원본 수정 없음) ---------------
    captured = []
    _orig_append = ev.append_results_csv

    def _capture(row, csv_path="evaluate_results.csv"):
        captured.append(dict(row))
        return "(evaluate_epoch.csv 로 보냄)"

    ev.append_results_csv = _capture

    total_new = 0
    for epoch in targets:
        todo = [s for s in scenarios if (epoch, s) not in done]
        if not todo:
            print(f"[epoch {epoch}] 이미 완료 — 건너뜀")
            continue
        ckpt = os.path.join(run_dir, f"pvtvae_epoch_{epoch}.pth")
        ev.find_latest_checkpoint_in = lambda _d, _c=ckpt: _c

        print(f"\n[epoch {epoch}] 평가 시작 — 시나리오 {todo}")
        captured.clear()
        ev.evaluate(scenarios=todo, limit=args.limit)

        rows = []
        for row in captured:
            r = {"epoch": epoch}
            r.update(row)
            rows.append(r)
        _append_rows(out_path, rows)
        total_new += len(rows)
        print(f"[epoch {epoch}] {len(rows)}행 기록 (누적 신규 {total_new})")

    ev.append_results_csv = _orig_append
    print("\n" + "=" * 74)
    print(f"완료 — {out_path} 에 신규 {total_new}행")
    print("=" * 74)


if __name__ == "__main__":
    main()
