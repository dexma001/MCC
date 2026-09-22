"""확장 아키텍처 평가 — AI_model/evaluate.py 를 '복사하지 않고' 재사용하는 얇은 드라이버.

[왜 복사하지 않는가]
  evaluate.py 는 71KB이고 지표 정의(cOKS, 3DPCK, intent_dyn, 경계 지표 …)가 전부 여기 있다.
  복사본을 두는 순간 두 파일이 갈라지고, 그 시점부터 소형/확장 런의 숫자는 **비교 불가능**해진다.
  대신 2026-09-08 체크포인트 전수 벤치마크가 쓴 것과 같은 방식을 쓴다:
  **모듈 전역만 바꿔 끼우고 evaluate.evaluate() 를 호출한다.** 프로젝트 파일은 건드리지 않는다.

  바꿔 끼우는 전역 4개:
    evaluate.MODEL_CLASS   <- TransformerDenoiserLargeCompat  (아키텍처)
    evaluate.RUN_TAG       <- AI_model_large/train_large.py 의 RUN_TAG (체크포인트 폴더 선택)
    evaluate.LAMBDA_*      <- 같은 파일의 λ (폴더 이름의 나머지 절반)
    evaluate.BETA_KL       <- 0.0

[작업 디렉터리]
  evaluate.py 는 `processed_motions_VMC/`, `checkpoints/`, `evaluate_results.csv` 를
  **상대 경로**로 찾는다. 그래서 이 드라이버는 시작할 때 프로젝트 루트로 chdir 한다.
  그래야 결과 행이 소형 런과 같은 `evaluate_results.csv` 에 쌓여 비교가 가능하다.

[CSV 오염 주의]
  evaluate.py 는 **실행할 때마다** 행을 추가한다. 배선만 확인하는 스모크 실행에는
  `--no-csv` 를 붙여라 (append_results_csv 를 몽키패치해 기록을 막는다).

실행 예:
    python AI_model_large/evaluate_large.py --limit 3 --no-csv        # 스모크
    python AI_model_large/evaluate_large.py                           # 전 시나리오 정식 평가
    python AI_model_large/evaluate_large.py --scenarios clean,transient
    python AI_model_large/evaluate_large.py --no-lowpass --no-projection   # 모델 단독
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_AI_MODEL_DIR = os.path.join(_ROOT, "AI_model")

# 두 폴더를 모두 얹는다. 이 폴더의 모듈명(models_large / train_large / evaluate_large)은
# AI_model 의 어떤 파일과도 겹치지 않으므로 **순서가 결과를 바꾸지 않는다** (의도된 설계).
for p in (_AI_MODEL_DIR, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    parser = argparse.ArgumentParser(description="확장 아키텍처(TransformerDenoiserLarge) 평가")
    parser.add_argument("--scenarios", type=str, default="",
                        help="쉼표 구분 (clean,legacy80,transient,persistent). 생략 시 전체")
    parser.add_argument("--limit", type=int, default=0, help="테스트 파일 수 제한 (0=전체)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="평가할 에폭 지정. 생략하면 가장 큰 에폭. "
                             "에폭을 늘려 학습했을 때 '마지막이 최선인지'를 확인하려면 필수다 "
                             "— evaluate.py 는 기본적으로 가장 큰 에폭을 자동 선택한다.")
    parser.add_argument("--no-csv", action="store_true",
                        help="evaluate_results.csv 에 기록하지 않는다 (스모크용)")
    parser.add_argument("--no-lowpass", action="store_true", help="저역통과 필터 끄기")
    parser.add_argument("--no-projection", action="store_true", help="사영 레이어 끄기")
    args = parser.parse_args()

    # evaluate.py 의 상대 경로 가정을 만족시킨다.
    os.chdir(_ROOT)

    import evaluate as ev
    import lowpass
    import projection
    import train_large                               # AI_model_large/train_large.py (상수만 읽는다)
    from models_large import TransformerDenoiserLargeCompat, arch_tag, LARGE_D_MODEL, \
        LARGE_NHEAD, LARGE_NUM_LAYERS, LARGE_DIM_FEEDFORWARD

    # --- 전역 바꿔 끼우기 ---------------------------------------------
    ev.MODEL_CLASS = TransformerDenoiserLargeCompat
    ev.RUN_TAG = train_large.RUN_TAG
    ev.LAMBDA_RECON = train_large.LAMBDA_RECON
    ev.LAMBDA_PHYS = train_large.LAMBDA_PHYS
    ev.BETA_KL = train_large.BETA_KL

    if args.no_lowpass:
        lowpass.LP_ENABLED = False
    if args.no_projection:
        projection.PROJ_ENABLED = False

    if args.epoch is not None:
        # evaluate.py 의 체크포인트 선택만 가로챈다 (원본 수정 없음).
        import os as _os
        _want = f"pvtvae_epoch_{args.epoch}.pth"

        def _pick(run_dir, _want=_want):
            p = _os.path.join(run_dir, _want)
            if not _os.path.exists(p):
                raise FileNotFoundError(f"해당 에폭 가중치가 없습니다: {p}")
            return p

        ev.find_latest_checkpoint_in = _pick

    if args.no_csv:
        _blocked = {"n": 0}

        def _noop(row, csv_path="evaluate_results.csv"):
            _blocked["n"] += 1
            return "(기록 안 함: --no-csv)"

        ev.append_results_csv = _noop

    print("=" * 72)
    print("확장 아키텍처 평가")
    print(f"  아키텍처   : d_model={LARGE_D_MODEL} nhead={LARGE_NHEAD} "
          f"layers={LARGE_NUM_LAYERS} ff={LARGE_DIM_FEEDFORWARD}  tag={arch_tag()}")
    print(f"  RUN_TAG    : {ev.RUN_TAG}")
    print(f"  에폭       : {'가장 큰 에폭(자동)' if args.epoch is None else args.epoch}")
    print(f"  λ          : recon={ev.LAMBDA_RECON} phys={ev.LAMBDA_PHYS} kl={ev.BETA_KL}")
    print(f"  대조군     : checkpoints/{train_large.BASELINE_RUN}")
    print(f"  저역통과   : {'OFF' if args.no_lowpass else lowpass.LP_MODE + f' (w={lowpass.LP_WINDOW})'}")
    print(f"  사영       : {'OFF' if args.no_projection else f'K={projection.PROJ_K} ω={projection.PROJ_OMEGA}'}")
    print(f"  CSV 기록   : {'안 함' if args.no_csv else 'evaluate_results.csv 에 추가'}")
    print(f"  작업 폴더  : {os.getcwd()}")
    print("=" * 72)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()] or None
    ev.evaluate(scenarios=scenarios, limit=args.limit)


if __name__ == "__main__":
    main()
