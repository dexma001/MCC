"""출시 전 자체 점검 — 방송 중에 처음 발견하면 안 되는 것들을 미리 터뜨린다. (T1~T10 런타임, T11~T15 Warudo 호환, T16 출력 시간축)

`claude_analysis/vtuber_live_deployment_gap_20260911.md` 의 P0/P2 항목을 실행 가능한
검사로 바꾼 것이다. **가장 중요한 것은 T6** — 진짜 관통을 주입해 사영의 그래디언트 경로를
실제로 태우는 검사다. 클린 입력만으로 하는 스모크는 위반 프레임이 0이라 그 경로에
진입조차 하지 않기 때문에, `inference_mode` 크래시 같은 결함을 **통과시켜 버린다.**

    python AI_Streaming/selftest.py
"""

import os
import random
import sys

import paths

import torch

import contract
import lowpass
import projection
from corrector import StreamingCorrector

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail))
        print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
    except AssertionError as e:
        RESULTS.append((False, name, str(e)))
        print(f"  ❌ {name} — {e}")
    except Exception as e:
        RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"  💥 {name} — {type(e).__name__}: {e}")


def main():
    print("=" * 74)
    print("AI_Streaming 자체 점검")
    print("=" * 74)

    physics = contract.make_physics("cpu")
    files = paths.test_motion_files(limit=4)
    assert files, "held-out 파일이 없습니다."
    motion = torch.load(files[0])
    clean_win = motion[:contract.SEQ_LEN].clone()          # [30, 87]

    import corruption
    cfg = corruption.make_cfg()
    rng = random.Random(20260707)
    soft_win = corruption.inject_persistent(
        clean_win, physics, cfg, rng, contract.COLLIDING_PAIRS)[0]

    ckpt = paths.checkpoint_path()
    from models import TransformerDenoiserCompat
    _model = TransformerDenoiserCompat(input_dim=87, output_dim=84, latent_dim=64)
    _model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    _model.eval()

    def _model_out(win):
        """모델 -> 필터 까지 통과시킨 (hips, 사영 전 quats, 필터까지 거친 quats)."""
        with torch.no_grad():
            out = _model(win.unsqueeze(0))[0][0]
        q_lp, _ = lowpass.lowpass_window(out[:, 3:].contiguous(), collect_stats=False)
        return out[:, :3], out[:, 3:].contiguous(), q_lp

    def pen_cm(win):
        d = physics.get_penetration_depths_from_quats(
            win[:, :3], win[:, 3:], contract.COLLIDING_PAIRS) * 100.0
        return float(d.max())

    # [중요] 검사용 손상 윈도우는 '찾아서' 쓴다 — 아무 윈도우나 쓰면 검사가 공허해진다.
    #   지속 주입(1~4cm)은 모델이 스스로 지워 버리는 경우가 있고, 고정 80° 주입조차
    #   포즈에 따라 관통을 만들지 못하는 윈도우가 있다(실측: held-out 5파일 중 3개가 0.00cm).
    #   그러면 사영 경로에 진입조차 하지 않아 T6/T8 이 **통과해 버린다**(처음 작성 때 실제로 그랬다).
    #   ⇒ '사영 직전에도 관통이 남아 있는' 윈도우를 탐색하고, 못 찾으면 검사를 실패시킨다.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "_ai_model_evaluate_for_selftest", os.path.join(paths.AI_MODEL_DIR, "evaluate.py"))
    _ev = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ev)

    def _scan_windows(limit_files=24):
        """held-out 파일들을 30프레임 청크로 잘라 80° 주입한 후보를 차례로 내놓는다."""
        for f in paths.test_motion_files(limit=limit_files):
            mo = torch.load(f)
            for s in range(0, mo.shape[0] - contract.SEQ_LEN + 1, contract.SEQ_LEN):
                yield (f"{os.path.basename(f)}[{s}:]",
                       _ev.inject_arm_collision(mo[s:s + contract.SEQ_LEN].clone(), angle_deg=80.0))

    def _find_window(pred, what):
        """pred(cand) 가 None 이 아닌 첫 윈도우를 찾는다. 못 찾으면 검사를 실패시킨다."""
        for name, cand in _scan_windows():
            got = pred(cand)
            if got is not None:
                return name, cand, got
        raise AssertionError(
            f"{what} 조건을 만족하는 윈도우를 held-out 24파일에서 찾지 못했습니다 — "
            f"이 검사는 아무것도 증명하지 못하므로 실패로 처리합니다.")

    def _pred_projection_needed(cand):
        """보정기가 실제로 내보내는 '마지막 프레임'에 사영 직전 관통이 남아 있는가.

        stride=1/emit_lag=0 에서 출력되는 프레임이 마지막 프레임이므로, 판정 기준도
        윈도우 최대가 아니라 마지막 프레임이어야 검사와 실제 출력이 일치한다.
        """
        hips, _, q_lp = _model_out(cand)
        pre = pen_cm(torch.cat([hips, q_lp], -1)[-1:])
        return pre if pre > 0.5 else None

    dirty_src, dirty_win, dirty_pre = _find_window(
        _pred_projection_needed, "사영 직전에 관통이 남는")
    print(f"  (T5~T7 용 손상 윈도우: {dirty_src} — 마지막 프레임의 사영 직전 관통 {dirty_pre:.2f}cm)")

    print("\n[A] 계약")

    def t1():
        problems, detail = contract.check_training_contract()
        assert not problems, "; ".join(problems)
        return (f"본 {detail['n_bones']}개, 페어 {len(detail['pairs'])}쌍, "
                f"신장 {detail['radii_stature_m']}m")
    check("T1 학습 코드와 계약 일치 (본 순서 · 충돌 페어)", t1)

    def t2():
        hips, quats = contract.unpack_frame(clean_win[0])
        again = contract.pack_frame(hips, quats)
        assert torch.equal(again, clean_win[0]), "pack(unpack(x)) != x"
        return "왕복 비트 동일"
    check("T2 pack/unpack 왕복", t2)

    def t3():
        problems = contract.validate_frame(clean_win[0], physics=physics)
        assert not problems, "; ".join(problems)
        # 일부러 망가뜨린 프레임은 반드시 걸려야 한다 (검사기가 죽어 있지 않은지 확인)
        bad = clean_win[0].clone()
        bad[3:] = bad[3:].roll(1)             # 쿼터니언 성분 순서를 어긋나게
        assert contract.validate_frame(bad, physics=physics), "망가진 프레임을 통과시켰습니다"
        return "정상 통과 / 오배선 검출 모두 확인"
    check("T3 프레임 계약 검사기 동작", t3)

    print("\n[B] 런타임 함정")

    ckpt = paths.checkpoint_path()

    def t4():
        try:
            with torch.inference_mode():
                StreamingCorrector(ckpt, physics=physics)
        except RuntimeError as e:
            assert "inference_mode" in str(e), f"다른 이유로 실패: {e}"
            return "가드가 명확한 메시지로 차단"
        raise AssertionError("inference_mode 안에서 생성이 허용됐습니다 (방송 중 크래시 위험)")
    check("T4 inference_mode 가드 (P0-2)", t4)

    corrector = StreamingCorrector(ckpt, physics=physics, validate_every=0)

    def t5():
        outs = []
        for i in range(contract.SEQ_LEN):
            o, info = corrector.push(dirty_win[i])
            outs.append((o, info))
        warm = [o for o, i in outs if i["stage"] == "warmup"]
        assert len(warm) == contract.SEQ_LEN - 1, f"워밍업 프레임 수 {len(warm)}"
        for k in range(contract.SEQ_LEN - 1):
            assert torch.equal(outs[k][0], dirty_win[k]), f"워밍업 {k}번 프레임이 변경됐습니다"
        return f"{len(warm)}프레임 원본 통과(비트 동일)"
    check("T5 워밍업 바이패스 (P0-3)", t5)

    def t6():
        # 마지막 프레임은 실제 계산을 탄다 = 사영 그래디언트 경로가 살아 있는지 확인
        c2 = StreamingCorrector(ckpt, physics=physics, validate_every=0)
        last = None
        for i in range(contract.SEQ_LEN):
            last, info = c2.push(dirty_win[i])
        assert info["stage"] == "compute", f"마지막 프레임 stage={info['stage']}"

        # --- 공허한 통과 방지: 사영 '직전' 상태에 실제로 관통이 남아 있어야 한다 ---
        hips, _, q_lp = _model_out(dirty_win)
        pre = pen_cm(torch.cat([hips, q_lp], -1)[-1:])
        assert pre > 0.5, (f"사영 직전 관통이 {pre:.3f}cm 뿐이라 이 검사는 아무것도 증명하지 "
                           f"못합니다 — 더 강한 손상을 쓰세요")
        assert info["k_used"] > 0, "사영이 한 번도 반복하지 않았습니다 (경로 미진입)"

        after = pen_cm(last.unsqueeze(0))
        assert after <= 0.25, f"관통이 남았습니다: {pre:.3f} -> {after:.3f} cm"
        return (f"입력 {pen_cm(dirty_win[-1:]):.2f}cm → 모델+필터 {pre:.2f}cm → "
                f"사영 후 {after:.3f}cm (반복 {info['k_used']}회)")
    check("T6 실제 관통 제거 — 사영 경로를 태우는 유일한 검사", t6)

    def t7():
        a = StreamingCorrector(ckpt, physics=physics, validate_every=0)
        b = StreamingCorrector(ckpt, physics=physics, validate_every=0)
        oa = ob = None
        for i in range(contract.SEQ_LEN):      # 다른 손상(지속 주입)으로도 확인한다
            oa, _ = a.push(soft_win[i])
            ob, _ = b.push(soft_win[i])
        assert torch.equal(oa, ob), "같은 입력에 다른 출력 (결정론 위반)"
        return "동일 입력 -> 비트 동일 출력"
    check("T7 결정론", t7)

    print("\n[C] 파이프라인 순서 (모델 -> 필터 -> 사영)")

    def t8():
        def both_orders(cand):
            hips, q, q_lp = _model_out(cand)
            q_right, _ = projection.project_window(       # 옳은 순서: 모델 -> 필터 -> 사영
                physics, hips, q_lp, contract.COLLIDING_PAIRS, collect_stats=False)
            q_proj, _ = projection.project_window(        # 뒤집은 순서: 모델 -> 사영 -> 필터
                physics, hips, q, contract.COLLIDING_PAIRS, collect_stats=False)
            q_wrong, _ = lowpass.lowpass_window(q_proj, collect_stats=False)
            right = pen_cm(torch.cat([hips, q_right], -1))
            wrong = pen_cm(torch.cat([hips, q_wrong], -1))
            return right, wrong

        def pred(cand):
            right, wrong = both_orders(cand)
            # 두 순서를 '가를 수 있는' 윈도우에서만 판정한다. 관통이 애초에 없는 윈도우는
            # 양쪽 다 0.000 이 나와 아무것도 증명하지 못한다.
            return (right, wrong) if (right < 0.25 and wrong > right + 0.1) else None

        name, _, (right, wrong) = _find_window(pred, "두 순서를 가를 수 있는")
        return (f"{name}: 필터→사영 {right:.3f}cm  vs  사영→필터 {wrong:.3f}cm "
                f"(뒤집으면 관통 부활)")
    check("T8 순서를 뒤집으면 관통이 되살아난다", t8)

    print("\n[D] 알려진 한계 재확인 (실패가 아니라 '기록')")

    def t9():
        with torch.no_grad():
            full = _model(clean_win.unsqueeze(0))[0][0, -1]
            short = _model(clean_win[:29].unsqueeze(0))[0][0, -1]
        p_full = physics.compute_global_pos_tensor(full[:3].reshape(1, 3), full[3:].reshape(1, 84))
        p_short = physics.compute_global_pos_tensor(short[:3].reshape(1, 3), short[3:].reshape(1, 84))
        diff = float((p_full - p_short).norm(dim=-1).mean() * 100.0)
        return f"29프레임 입력은 30프레임과 {diff:.3f}cm 다른 답을 낸다 (에러 없이!)"
    check("T9 S<30 은 조용히 다른 답 (그래서 워밍업 바이패스가 필요하다)", t9)

    def t10():
        try:
            with torch.no_grad():
                _model(torch.cat([clean_win, clean_win[:1]], 0).unsqueeze(0))
        except RuntimeError:
            return "31프레임 입력은 RuntimeError (예상된 동작)"
        raise AssertionError("31프레임이 통과했습니다 — 위치 임베딩 가정이 바뀌었는지 확인 필요")
    check("T10 S>30 은 즉시 실패", t10)

    print("\n[E] Warudo 호환 (VMC 프로토콜 · VRM 정규화 본)")

    import retarget as rt
    import sinks

    def t11():
        probs = contract.check_vmc_names()
        assert not probs, "; ".join(probs)
        return "21본 전부 Unity HumanBodyBones 철자"
    check("T11 본 이름이 Unity HumanBodyBones 철자인가", t11)

    retg = rt.Retarget()

    def t12():
        # 리그의 T-포즈 로컬은 단위가 아니다 — 변환 없이 보내면 뒤틀린다는 사실 자체를 기록
        tpose = retg.rig_tpose_frame()
        raw_ang = rt.qangle_deg(tpose[3:].reshape(21, 4))
        raw_ang = torch.minimum(raw_ang, 360 - raw_ang)
        v = retg.rig_to_vrm(tpose)
        vrm_ang = rt.qangle_deg(v[3:].reshape(21, 4))
        vrm_ang = torch.minimum(vrm_ang, 360 - vrm_ang)
        assert float(vrm_ang.max()) < 1e-3, f"T-포즈가 VRM 단위 회전으로 안 감: max {float(vrm_ang.max()):.4f}°"
        assert float(raw_ang.max()) > 90, "리그 T-포즈 로컬이 거의 단위 — rig_tpose.json 이 의심스럽다"
        # 왕복
        f = clean_win[0]
        back = retg.vrm_to_rig(retg.rig_to_vrm(f))
        d = (back[3:].reshape(21, 4) * f[3:].reshape(21, 4)).sum(-1).abs()
        assert float((1 - d).max()) < 1e-5, "vrm_to_rig(rig_to_vrm(x)) != x"
        return (f"리그 T-포즈 로컬 최대 {float(raw_ang.max()):.0f}° (그대로 보내면 뒤틀림) → "
                f"VRM {float(vrm_ang.max()):.1e}°; 왕복 일치")
    check("T12 리타깃: 리그 T-포즈 → VRM 단위 회전, 왕복 동일", t12)

    def t13():
        # 변환이 모든 관절의 월드 위치를 보존하는가 (VRM 오프셋 리그로 FK)
        off = retg.vrm_offsets(physics)
        m = torch.load(files[1])[:60]
        v = retg.rig_to_vrm(m)
        P_rig = physics.compute_global_pos_tensor(m[:, :3], m[:, 3:])
        P_vrm, _ = rt.fk_with_offsets(v[:, :3], v[:, 3:].reshape(-1, 21, 4), off)
        P_vrm = torch.stack([P_vrm[b] for b in contract.BONE_ORDER], dim=1)
        err = float((P_rig - P_vrm).norm(dim=-1).max() * 100)
        assert err < 0.01, f"위치 보존 실패 {err:.4f} cm"
        # 팔이 T-포즈에서 수평인가 (rig_tpose.json 의 기하)
        t = retg.rig_tpose_frame()
        gp = physics.forward_kinematics(t[:3].reshape(1, 3), t[3:].reshape(1, 84))
        arm = (gp['LeftHand'] - gp['LeftShoulder']).reshape(3)
        cos = float(arm @ torch.tensor([-1.0, 0, 0]) / arm.norm())
        assert cos > 0.999, f"T-포즈 왼팔이 -X 가 아님 (cos {cos:.4f})"
        return f"관절 위치 최대 오차 {err:.5f} cm, T-포즈 팔 수평"
    check("T13 리타깃이 관절 위치를 보존한다 (VRM 리그 FK 일치)", t13)

    def t14():
        sink = sinks.VmcOscSink(retarget=retg)                  # DRY-RUN
        msgs = sink.build(clean_win[0])
        addrs = [a for a, _ in msgs]
        assert addrs[0] == "/VMC/Ext/OK" and addrs[-1] == "/VMC/Ext/T", f"OK/T 순서: {addrs[0]} .. {addrs[-1]}"
        bones = [v[0] for a, v in msgs if a == "/VMC/Ext/Bone/Pos"]
        assert sorted(bones) == sorted(contract.BONE_ORDER), "21본 누락/중복"
        hips = [v for a, v in msgs if a == "/VMC/Ext/Bone/Pos" and v[0] == "Hips"][0]
        assert len(hips) == 8 and all(isinstance(x, float) for x in hips[1:]), f"Hips 인자 형식 {hips}"
        others = [v for a, v in msgs if a == "/VMC/Ext/Bone/Pos" and v[0] != "Hips"]
        assert all(v[1:4] == [0.0, 0.0, 0.0] for v in others), "Hips 외 본에 위치가 실림"
        for v in [v for a, v in msgs if a == "/VMC/Ext/Bone/Pos"]:
            n = sum(x * x for x in v[4:8]) ** 0.5
            assert abs(n - 1) < 1e-4, f"{v[0]} 쿼터니언 노름 {n}"
        # 보낸 회전이 VRM 규약인가: 리그 값과 달라야 한다
        raw = sinks.VmcOscSink(retarget=False).build(clean_win[0])
        rq = [v[4:8] for a, v in raw if a == "/VMC/Ext/Bone/Pos" and v[0] == "Spine"][0]
        vq = [v[4:8] for a, v in msgs if a == "/VMC/Ext/Bone/Pos" and v[0] == "Spine"][0]
        assert abs(rq[1]) > 0.5 and abs(vq[1]) < 0.3, f"Spine 리그 {rq} vs VRM {vq} — 변환이 적용되지 않음"
        b = sinks.VmcOscSink.bundle(msgs)                       # python-osc 번들 직렬화
        assert len(b.dgram) > 500, "번들 직렬화 실패"
        return f"메시지 {len(msgs)}개 (OK·root·21본·T), 번들 {len(b.dgram)}B, Spine 리그 y={rq[1]:.2f}→VRM y={vq[1]:.2f}"
    check("T14 VMC 패킷 형식 (OK/Root/21본/T, Hips 만 위치, 단위 쿼터니언, 변환 적용)", t14)

    def t15():
        import vmc_bridge
        c = StreamingCorrector(ckpt, physics=physics, validate_every=0)
        ok = vmc_bridge.loopback_test(c, retg, physics, port_in=39565, port_out=39566, n=45, yaw_deg=37.0)
        assert ok
        return ("UDP 소켓 왕복 + 트래커 yaw 37° 정규화·복원 + Hips 절대 위치 복원 + "
                "손가락/표정 pass-through + 21본 이중 전송 없음 + "
                "합성 관통 모션(전 프레임 4쌍 관통) → 수신 보정 프레임 관통 0 + 보정 예외 0")
    check("T15 소켓 루프백 (트래커 → 브리지 → 수신) — Warudo 없이 전 구간", t15)

    print("\n[F] 출력 시간축")

    def t16():
        # Hips x 에 프레임 번호를 새긴다. 보정 경로는 Hips 를 입력 그대로 싣고 나가므로
        # 출력의 Hips x 가 곧 '그 출력이 몇 번 입력 프레임인가'다.
        base = clean_win[0].clone()
        lines = []
        for stride, lag, toggle in [(1, 0, False), (1, 2, False), (5, 3, True), (30, 0, False)]:
            c = StreamingCorrector(ckpt, physics=physics, validate_every=0, stride=stride, emit_lag=lag,
                                   lp_enabled=False, proj_enabled=False)
            seq, stages = [], []
            for i in range(75):
                if toggle and i == 40:
                    c.bypass = True
                if toggle and i == 47:
                    c.bypass = False
                f = base.clone()
                f[0] = float(i)
                o, info = c.push(f)
                if o is not None:
                    seq.append(int(round(float(o[0]))))
                    stages.append(info["emitted_stage"])
            d = c.delay
            assert d == stride - 1 + lag, f"delay {d} != stride-1+lag"
            assert seq == list(range(75 - d)), (f"stride={stride} lag={lag}: 출력 번호가 연속이 아닙니다 "
                                                f"(앞부분 {seq[:6]}, 29 근처 {seq[26:33]})")
            n_cor = stages.count("corrected")
            assert n_cor >= 30, f"stride={stride} lag={lag}: 보정 프레임이 {n_cor}개뿐 — 검사가 공허합니다"
            lines.append(f"s{stride}/lag{lag}{'+bypass' if toggle else ''}: 지연 {d}f, 보정 {n_cor}")
        return "출력 번호 0,1,2,… 연속(뒤로 점프 없음) — " + "; ".join(lines)
    check("T16 stride/emit_lag 에서도 출력 시간이 연속이고 지연이 일정하다", t16)

    n_fail = sum(1 for ok, _, _ in RESULTS if not ok)
    print("\n" + "=" * 74)
    print(f"결과: {len(RESULTS) - n_fail}/{len(RESULTS)} 통과")
    if n_fail:
        print("❌ 실패 항목이 있습니다 — 송신을 붙이기 전에 해결하세요.")
    else:
        print("✅ 전 항목 통과. 다음 단계: run_sim.py 로 지연·품질을 측정하세요.")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
