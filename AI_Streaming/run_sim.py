"""라이브 시뮬레이션 — 학습한 모델을 '실시간 환경'에서 돌리고 판정한다.

실제 송신 전에 반드시 통과시켜야 하는 단계다. held-out 모션을 한 프레임씩 흘려보내며
  (1) 지연이 프레임 예산 안에 들어오는가
  (2) 관통이 실제로 제거되는가
  (3) 멀쩡한 입력을 건드리지 않는가 (do-no-harm)
를 같은 실행에서 측정한다.

실행 예:
    python AI_Streaming/run_sim.py                                  # clean, 600프레임
    python AI_Streaming/run_sim.py --scenario persistent --frames 900
    python AI_Streaming/run_sim.py --scenario legacy80 --no-projection   # 사영 끄고 대조
    python AI_Streaming/run_sim.py --pace                            # 실제 30fps 시계로
    python AI_Streaming/run_sim.py --stride 30                       # 타일링(경계 팝 확인)
    python AI_Streaming/run_sim.py --osc-dry-run                     # VMC 패킷 생성까지
"""

import argparse
import os
import sys

import paths

import torch

import contract
import sources
import sinks
from corrector import StreamingCorrector


def _fk_positions(physics, frames87):
    """[N, 87] -> [N, 21, 3] 월드 좌표 (m)."""
    return physics.compute_global_pos_tensor(frames87[:, :3], frames87[:, 3:])


def _pen_stats(physics, frames87, pairs):
    """4쌍 관통 통계 (cm). (max, mean_over_colliding, colliding_frame_pct)"""
    d = physics.get_penetration_depths_from_quats(
        frames87[:, :3], frames87[:, 3:], pairs) * 100.0      # [N, 4] cm
    per_frame = d.max(dim=-1).values
    colliding = per_frame > 0
    return (float(per_frame.max()),
            float(per_frame[colliding].mean()) if colliding.any() else 0.0,
            100.0 * float(colliding.float().mean()))


def _jitter(pos):
    """mean||accel|| (cm/frame^2). evaluate.py 의 지터 정의와 같은 형태."""
    if pos.shape[0] < 3:
        return 0.0
    acc = pos[2:] - 2 * pos[1:-1] + pos[:-2]
    return float(acc.norm(dim=-1).mean() * 100.0)


def main():
    ap = argparse.ArgumentParser(description="라이브 스트리밍 시뮬레이션")
    ap.add_argument("--run", default=paths.DEFAULT_RUN, help="체크포인트 run 폴더 이름")
    ap.add_argument("--epoch", type=int, default=None, help="에폭 지정 (기본: 가장 큰 에폭)")
    ap.add_argument("--scenario", default="clean",
                    choices=["clean", "transient", "persistent", "legacy80"])
    ap.add_argument("--frames", type=int, default=600, help="시뮬레이션 프레임 수 (30fps 기준 20초)")
    ap.add_argument("--files", type=int, default=8, help="사용할 held-out 파일 수")
    ap.add_argument("--device", default="cpu",
                    help="cpu 권장 — 단일 윈도우에서 CUDA 는 3~6배 느리다(실측)")
    ap.add_argument("--stride", type=int, default=1, help="1=슬라이딩, 30=타일링")
    ap.add_argument("--emit-lag", type=int, default=0, help="0=최신 프레임, 2=영위상 필터 안전")
    ap.add_argument("--budget-ms", type=float, default=1000.0 / 30, help="프레임당 시간 예산")
    ap.add_argument("--no-lowpass", action="store_true")
    ap.add_argument("--no-projection", action="store_true")
    ap.add_argument("--no-adaptive-k", action="store_true", help="예산 기반 K 축소 끄기")
    ap.add_argument("--pace", action="store_true", help="실제 시계에 맞춰 30fps 로 흘린다")
    ap.add_argument("--validate-every", type=int, default=90, help="N프레임마다 입력 계약 검사")
    ap.add_argument("--osc-dry-run", action="store_true", help="VMC 패킷 생성까지 (전송 안 함)")
    ap.add_argument("--large", action="store_true",
                    help="AI_model_large 의 확장 아키텍처로 로드")
    args = ap.parse_args()

    ckpt = paths.checkpoint_path(args.run, args.epoch)
    physics = contract.make_physics(args.device)

    model_class = None
    if args.large:
        sys.path.insert(0, os.path.join(paths.ROOT, "AI_model_large"))
        from models_large import TransformerDenoiserLargeCompat as model_class  # noqa: N813

    corrector = StreamingCorrector(
        ckpt, device=args.device, model_class=model_class,
        stride=args.stride, emit_lag=args.emit_lag, budget_ms=args.budget_ms,
        lp_enabled=not args.no_lowpass, proj_enabled=not args.no_projection,
        adaptive_k=not args.no_adaptive_k, validate_every=args.validate_every,
        physics=physics)

    files = paths.test_motion_files(limit=args.files)
    src = sources.ReplaySource(files, scenario=args.scenario, pace=args.pace,
                               physics=physics, max_frames=args.frames)

    rec = sinks.RecordSink()
    osc = sinks.VmcOscSink() if args.osc_dry_run else None

    print("=" * 74)
    print("라이브 시뮬레이션")
    print(f"  체크포인트 : {os.path.relpath(ckpt, paths.ROOT)}")
    print(f"  아키텍처   : {'AI_model_large (확장)' if args.large else 'AI_model (기본)'}")
    print(f"  시나리오   : {args.scenario} — {sources.describe_scenarios()[args.scenario]}")
    print(f"  파이프라인 : 모델"
          f"{'' if args.no_lowpass else ' -> 저역통과'}"
          f"{'' if args.no_projection else ' -> 사영'}")
    print(f"  윈도우     : seq_len=30 stride={args.stride} emit_lag={args.emit_lag}")
    print(f"  예산       : {args.budget_ms:.2f} ms/frame   장치: {args.device}")
    print(f"  입력       : held-out {len(files)}파일, {args.frames}프레임 "
          f"({'실시간 페이싱' if args.pace else '최대 속도'})")
    print("=" * 74)

    inputs, cleans, stages = [], [], []
    for inp, clean in src:
        out, info = corrector.push(inp)
        if out is not None:
            rec.send(out)
            stages.append(info["emitted_stage"])
            if osc is not None:
                osc.send(out)
        inputs.append(inp)
        cleans.append(clean)

    # 출력 k 번째 = 입력 k 번째 프레임이다 (보정기가 지연 delay 동안 None 을 내고, 그 뒤로는
    # 항상 '입력 번호 - delay' 를 내보내므로 None 을 뺀 출력 열은 입력 열과 번호가 맞는다).
    # 끝의 delay 프레임은 아직 출력되지 않았으므로 짧은 쪽에 맞춘다.
    n = min(len(rec.frames), len(inputs))
    if n < 3:
        print("❌ 출력 프레임이 너무 적습니다.")
        return
    out_t = rec.tensor()[:n]
    in_t = torch.stack(inputs[:n], dim=0)
    cl_t = torch.stack(cleans[:n], dim=0)

    s = corrector.summary()

    print("\n[1] 지연 (판정 기준은 평균이 아니라 p95/max)")
    print(f"  프레임당 총 시간   p50 {s['ms_p50']:7.3f} / p95 {s['ms_p95']:7.3f} / "
          f"max {s['ms_max']:7.3f} ms   (예산 {s['budget_ms']:.2f})")
    print(f"  예산 초과 프레임   {s['over_budget_frames']}/{s['frames_in']} "
          f"({s['over_budget_pct']:.2f}%)")
    print(f"  단계별 평균        모델 {s['ms_model_mean']:.3f} / 필터 {s['ms_lp_mean']:.3f} / "
          f"사영 {s['ms_proj_mean']:.3f} ms   (사영 최대 {s['ms_proj_max']:.3f})")
    print(f"  사영 반복          최대 {s['proj_iters_max']}회, 예산으로 깎인 횟수 {s['k_capped']}")
    print(f"  워밍업 통과        {s['warmup_frames']}프레임 (버퍼 30 채우는 동안 원본 통과)")
    print(f"  출력 지연          {s['delay_frames']}프레임 = stride-1 + emit_lag "
          f"(≈ {s['delay_frames'] * 1000.0 / contract.FPS:.0f} ms, 처리 시간 별도)"
          + (f"   보정본 없는 원본 통과 {s['passthrough_frames']}프레임" if s['passthrough_frames'] else ""))

    print("\n[2] 관통 — 4쌍 (사영이 보장하는 범위)")
    pin = _pen_stats(physics, in_t, contract.COLLIDING_PAIRS)
    pout = _pen_stats(physics, out_t, contract.COLLIDING_PAIRS)
    print(f"  입력   max {pin[0]:6.3f} cm / 충돌프레임 {pin[2]:6.2f}%")
    print(f"  출력   max {pout[0]:6.3f} cm / 충돌프레임 {pout[2]:6.2f}%")
    if pin[0] > 0:
        print(f"  최악 관통 제거율  {100 * (1 - pout[0] / pin[0]):6.2f}%")
    print(f"  사영이 손댄 프레임 누계 {s['proj_frames_touched']}")
    # [중요] 출력에 남은 관통을 '보정 실패'로 읽으면 틀린다. 스트림 시작 1초(30프레임)는
    #   버퍼가 찰 때까지 원본을 그대로 통과시키므로, 그 구간의 관통은 파이프라인이
    #   손대지 않은 것이다. 출처별로 갈라서 본다.
    st = torch.tensor([0 if x == "corrected" else 1 for x in stages[:n]], dtype=torch.bool)
    if st.any():
        pby = _pen_stats(physics, out_t[st], contract.COLLIDING_PAIRS)
        print(f"  └ 원본 통과분({int(st.sum())}프레임, 워밍업/패닉)  max {pby[0]:6.3f} cm "
              f"/ 충돌프레임 {pby[2]:6.2f}%")
    if (~st).any():
        pcor = _pen_stats(physics, out_t[~st], contract.COLLIDING_PAIRS)
        print(f"  └ 보정된 프레임({int((~st).sum())}프레임)            max {pcor[0]:6.3f} cm "
              f"/ 충돌프레임 {pcor[2]:6.2f}%   ← 파이프라인의 실제 성적")

    print("\n[3] 충실도 / do-no-harm")
    pos_in = _fk_positions(physics, in_t)
    pos_out = _fk_positions(physics, out_t)
    pos_cl = _fk_positions(physics, cl_t)
    move = float((pos_out - pos_in).norm(dim=-1).mean() * 100.0)
    move_hand = float(torch.stack([
        (pos_out[:, contract.BONE_INDEX['LeftHand']] - pos_in[:, contract.BONE_INDEX['LeftHand']]).norm(dim=-1),
        (pos_out[:, contract.BONE_INDEX['RightHand']] - pos_in[:, contract.BONE_INDEX['RightHand']]).norm(dim=-1),
    ]).mean() * 100.0)
    mpjpe = float((pos_out - pos_cl).norm(dim=-1).mean() * 100.0)
    mpjpe_in = float((pos_in - pos_cl).norm(dim=-1).mean() * 100.0)
    print(f"  입력 대비 이동량   전 관절 {move:6.3f} cm / 손 {move_hand:6.3f} cm")
    print(f"  클린 대비 오차     입력 {mpjpe_in:6.3f} -> 출력 {mpjpe:6.3f} cm "
          f"({'개선' if mpjpe < mpjpe_in else '악화'})")
    print(f"  지터(mean|accel|)  입력 {_jitter(pos_in):6.3f} -> 출력 {_jitter(pos_out):6.3f} "
          f"/ 클린 {_jitter(pos_cl):6.3f}   [연속 스트림 기준 — CSV 행과 직접 비교 불가]")
    if args.scenario == "clean":
        print("  ⚠️ clean 시나리오에서 '입력 대비 이동량'이 do-no-harm 지표다. "
              "0에 가까울수록 좋다.")

    print("\n[4] 계약")
    print(f"  입력 계약 위반 검출 {s['contract_problems']}건 "
          f"({args.validate_every}프레임마다 검사)")
    print(f"  프레임 in/out       {s['frames_in']} / {s['frames_out']}")

    if osc is not None:
        print("\n[5] 송신 (DRY-RUN)")
        print("  " + osc.report().replace("\n", "\n  "))

    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
