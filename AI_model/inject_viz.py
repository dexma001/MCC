"""
손상 주입(클리핑) 시각화 — transient / persistent 가 '어떻게' 들어가는지 gif로 만든다.

학습(train.py)·평가(evaluate.py)·데모(demo_maker.py)와 **동일한 corruption.py 주입기**를
호출하므로, 여기서 보이는 것이 실제 학습 입력에 들어가는 손상 그 자체다.

화면 구성 (3분할):
  왼쪽  3D : Clean(원본)           — 주입 대상 본을 초록으로 강조
  가운데 3D: Injected(주입 결과)   — 침투 깊이에 비례해 빨강 강도/캡슐 알파 상승
  아래 그래프: 프레임별 최대 침투 깊이(cm) + 주입 구간 음영 + 현재 프레임 커서
             → 두 유형의 '시간적 서명' 차이가 한눈에 보인다.
                transient  = 봉우리 하나 (sin-ramp: 들어왔다 빠짐)
                persistent = 구간 내내 유지되는 고원(plateau)

실행 방법은 motion_viz.py 와 같은 두 가지를 모두 지원한다.

  (A) 소스 설정 + VS Code 실행 — 아래 '소스 설정' 블록을 고치고 Run(Ctrl+F5).
  (B) 명령줄 인자 (소스 설정보다 우선):
        python AI_model/inject_viz.py                    # 소스 설정대로
        python AI_model/inject_viz.py both               # 두 유형 gif 모두 생성
        python AI_model/inject_viz.py transient
        python AI_model/inject_viz.py persistent --seed 1234
        python AI_model/inject_viz.py both --out demo_results --show

  .vscode/launch.json 에 F5 구성도 등록되어 있다.

⚠️ 좌표 규약: 주입/물리(FK)는 Unity(Y-Up) 텐서에서 그대로 수행하고, 화면에 그릴 때만
   motion_viz 의 어댑터로 Z-Up 변환한다 (학습 경로와 동일한 규약).
"""
import argparse
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

import corruption
import motion_viz as mv
from dataset_pipeline import BONE_MAP, BONE_RADII, PARENTS, get_split_files
from physics_module import DifferentiablePhysics
from train import COLLIDING_PAIRS

# =====================================================================
# 소스 설정 — 인자 없이 Run(Ctrl+F5)할 때 쓰이는 값
# =====================================================================
INJECT_MODE = "both"        # "transient" / "persistent" / "both"
INJECT_SEED = 1234         # 파일·주입 추첨 시드 (고정하면 같은 gif가 재현된다)
OUT_DIR = "demo_results"    # gif 저장 폴더
SHOW_WINDOW = False         # True면 저장 후 창도 띄운다
TARGET_FILE = ""            # 특정 .pt 고정 (빈 문자열 = held-out 테스트셋에서 추첨)
SEQ_LEN = 30                # 학습 윈도우와 동일 (바꾸면 학습 분포와 달라진다)
FPS = 10                    # gif 재생 속도 (느릴수록 관찰하기 쉽다)
MIN_DEPTH_CM = 1.0          # 이만큼도 안 뚫리면 다른 파일로 재추첨 (가시성 확보)
MAX_FILE_TRIES = 30

INJECT_MODES = ("transient", "persistent", "both")


def _use_korean_font():
    """
    한글 라벨이 네모(두부)로 깨지지 않게 한글 폰트를 지정한다.
    설치된 것이 없으면 조용히 기본 폰트를 쓰되(경고만), 그림 생성은 계속한다.
    (matplotlib 기본 DejaVu Sans에는 한글 글리프가 없다.)
    """
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Malgun Gothic", "NanumGothic", "Noto Sans KR", "AppleGothic", "Gulim"):
        if cand in available:
            plt.rcParams['font.family'] = cand
            plt.rcParams['axes.unicode_minus'] = False   # 한글 폰트의 마이너스 깨짐 방지
            return cand
    print("⚠️ 한글 폰트를 찾지 못했습니다 — 라벨이 네모로 보일 수 있습니다.")
    return None

# 주입 '형태' 파라미터는 학습/평가와 같은 설계 기본값을 쓴다 — 데모용으로 왜곡하지 않는다.
INJECT_CFG = corruption.make_cfg()

_INJECTORS = {
    'transient': corruption.inject_transient,
    'persistent': corruption.inject_persistent,
}

_DESC = {
    'transient': "일시적(글리치): sin-ramp 으로 들어왔다 빠진다",
    'persistent': "지속적(의도/체형 불일치): 구간 내내 관통이 유지된다",
}


def _pick_and_inject(kind, physics, seed, target_file="", pt_dir=None):
    """
    held-out 테스트 파일에서 추첨해 주입한다. 눈에 보일 만큼 뚫릴 때까지 재추첨.
    반환: (파일경로, clean [S,87], injected [S,87], meta)

    demo_maker.py 와 같은 규약: 채택 판정은 meta['collided']가 아니라 '깊이'로 한다
    (collided는 '주입기 내부 임계 통과' 플래그라 실제 충돌 여부와 다르다).
    """
    pt_dir = pt_dir or mv.PROCESSED_PT_DIR
    if target_file:
        cands = [target_file]
    else:
        cands = get_split_files(pt_dir, split='test')
        random.Random(seed).shuffle(cands)

    inject = _INJECTORS[kind]
    best = None
    for i, fpath in enumerate(cands[:MAX_FILE_TRIES]):
        full = torch.load(fpath)
        if full.shape[0] < SEQ_LEN:
            continue
        clean = full[:SEQ_LEN].clone()
        rng = random.Random(seed + 10007 * i)
        injected, meta = inject(clean, physics, INJECT_CFG, rng, COLLIDING_PAIRS)
        depth = meta.get('max_depth_cm', 0.0)
        if depth >= MIN_DEPTH_CM:
            return fpath, clean, injected, meta
        if depth > 0.0 and (best is None or depth > best[3].get('max_depth_cm', 0.0)):
            best = (fpath, clean, injected, meta)
    if best is None:
        raise RuntimeError(f"{MAX_FILE_TRIES}개 파일 모두 주입이 충돌을 만들지 못했습니다. "
                           f"--seed 를 바꿔 다시 시도하세요.")
    print(f"  ⚠️ 목표 깊이 {MIN_DEPTH_CM}cm 미달 — 가장 깊은 후보"
          f"({best[3].get('max_depth_cm', 0.0):.2f}cm)로 진행합니다.")
    return best


def _frame_depths(motion_87, physics):
    """프레임별 최대 침투 깊이(cm) [F] — 물리 엔진 기준(시각화 색과 같은 4페어)."""
    dep = physics.get_penetration_depths_from_quats(
        motion_87[:, :3], motion_87[:, 3:], COLLIDING_PAIRS) * 100.0
    return dep.max(dim=1).values


def make_injection_gif(kind, out_path, seed=INJECT_SEED, target_file="", show=False):
    """kind('transient'/'persistent') 주입을 Clean/Injected + 깊이 그래프 gif로 저장."""
    _use_korean_font()
    physics = DifferentiablePhysics(PARENTS, BONE_RADII)
    offsets_py = mv.convert_offsets_to_python(mv._load_offsets_or_exit())

    fpath, clean, injected, meta = _pick_and_inject(kind, physics, seed, target_file)
    f0, f1 = meta.get('frames', (0, SEQ_LEN - 1))
    bone = meta.get('bone', '?')

    depths = _frame_depths(injected, physics)          # [F] cm
    seg = depths[f0:f1 + 1]
    cov = float((seg > 1e-4).float().mean()) if len(seg) else 0.0

    print(f"[{kind}] {os.path.basename(fpath)}")
    print(f"  주입 본: {bone} | 구간: {f0}~{f1}"
          + (f" ({meta['seg']})" if 'seg' in meta else ""))
    print(f"  깊이: 최대 {float(depths.max()):.2f}cm | 구간 중앙값 "
          f"{float(seg.median()) if len(seg) else 0.0:.2f}cm | coverage {cov:.2f}")

    clean_py = mv.convert_unity_to_python_tensor(clean)
    inj_py = mv.convert_unity_to_python_tensor(injected)
    bones_to_draw = mv._bones_to_draw()
    pair_thr = [BONE_RADII[c1] + BONE_RADII[c2] for (_, _, _, c1, _, c2) in mv.TEST_PAIRS]

    # 주입된 본의 '하위 체인'을 강조한다 — 회전은 자손 전체를 움직이기 때문.
    highlight = set()
    if bone in BONE_MAP:
        stack = [bone]
        while stack:
            b = stack.pop()
            highlight.add(b)
            stack += [c for c, p in PARENTS.items() if p == b]

    # ---- 레이아웃: 3D 두 개(위, 주역) + 깊이 그래프(아래, 조역) ----
    # 구판은 add_subplot(2,2,·)/(2,1,2)라 그래프가 세로 절반을 차지해 정작 비교해야 할
    # 두 애니메이션이 작아졌다. GridSpec으로 3D:그래프 = 3.4:1 로 주고, 3D 패널은
    # 서로 붙여(wspace 축소) 최대한 크게 잡는다.
    fig = plt.figure(figsize=(13, 8.5))
    fig.suptitle(f"Clipping Injection — {kind.upper()}", fontsize=15, fontweight='bold')
    gs = fig.add_gridspec(2, 2, height_ratios=[3.4, 1.0],
                          left=0.055, right=0.985, top=0.93, bottom=0.075,
                          wspace=0.02, hspace=0.12)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    axg = fig.add_subplot(gs[1, :])

    ax1.set_title("Clean (원본)", fontsize=11)
    ax2.set_title(f"Injected — {bone}", fontsize=11)

    # 축 범위를 '실제 뼈대가 차지하는 공간'에 맞춘다. 고정 범위(±0.8, 0~1.8)를 쓰면
    # 루트가 원점에서 멀거나 모션이 작을 때 캐릭터가 구석에 작게 박혀 보인다.
    # 두 패널은 같은 범위를 공유해야 Clean/Injected 차이가 크기 왜곡 없이 비교된다.
    allpos = torch.stack([mv.get_pos_tensor(clean_py[k], offsets_py) for k in range(len(clean_py))]
                         + [mv.get_pos_tensor(inj_py[k], offsets_py) for k in range(len(inj_py))])
    lo_xyz = allpos.reshape(-1, 3).min(dim=0).values
    hi_xyz = allpos.reshape(-1, 3).max(dim=0).values
    center = ((lo_xyz + hi_xyz) / 2.0).tolist()
    half = float((hi_xyz - lo_xyz).max()) * 0.52 + 0.03   # 여유 ~4% (구판 20%는 과했다)
    for ax in (ax1, ax2):
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.view_init(elev=15, azim=45)
        ax.tick_params(labelsize=6, pad=0)
        # 3D 축의 기본 여백이 커서 뼈대가 실제보다 작아 보인다. 적당히 확대하되
        # 1.2를 넘기면 머리/발이 패널 밖으로 잘리므로 여기서 멈춘다 (실측 확인).
        try:
            ax.set_box_aspect((1, 1, 1), zoom=1.18)
        except TypeError:      # 구버전 matplotlib에는 zoom 인자가 없다
            ax.set_box_aspect((1, 1, 1))

    lines_clean = [ax1.plot([], [], [], 'o-', lw=2.0, color='lightgray')[0]
                   for _ in bones_to_draw]
    lines_inj = [ax2.plot([], [], [], 'o-', lw=2.0, color='dimgray')[0]
                 for _ in bones_to_draw]
    caps_inj = []
    for _p, child in bones_to_draw:
        caps_inj.append(ax2.plot([], [], [], '-', lw=BONE_RADII.get(child, 0.05) * 400.0,
                                 color='dodgerblue', alpha=0.15)[0])

    txt = ax2.text2D(0.02, 0.98, "", transform=ax2.transAxes,
                     fontsize=11, va='top', fontweight='bold')

    # ---- 깊이 그래프: 두 유형의 시간적 서명을 보여주는 핵심 패널 ----
    F = injected.shape[0]
    axg.plot(range(F), depths.tolist(), '-o', ms=3, lw=1.6, color='crimson',
             label='최대 침투 깊이')
    axg.axvspan(f0 - 0.5, f1 + 0.5, color='orange', alpha=0.15,
                label=f'주입 구간 ({f0}~{f1})')
    axg.axhspan(1.0, 4.0, color='green', alpha=0.08, label='목표 1~4cm')
    cursor = axg.axvline(0, color='black', lw=1.8)
    axg.set_xlim(-0.5, F - 0.5)
    axg.set_ylim(0, max(4.5, float(depths.max()) * 1.15))
    # 패널이 낮아졌으므로 라벨/제목을 압축한다. 제목은 축 안쪽에 얹어 세로 공간을 아낀다.
    axg.set_xlabel("frame", fontsize=8, labelpad=1)
    axg.set_ylabel("penetration (cm)", fontsize=8, labelpad=2)
    axg.tick_params(labelsize=7, pad=1)
    axg.text(0.012, 0.90, _DESC[kind], transform=axg.transAxes,
             fontsize=9, va='top', fontweight='bold', color='#333333')
    axg.grid(alpha=0.3)
    axg.legend(loc='upper right', fontsize=7, ncol=3, framealpha=0.85,
               borderpad=0.3, handlelength=1.4, columnspacing=1.0)

    def draw_panel(lines, caps, pos_np, pen_by_bone):
        for i, (parent, child) in enumerate(bones_to_draw):
            p, c = pos_np[BONE_MAP[parent]], pos_np[BONE_MAP[child]]
            lines[i].set_data([p[0], c[0]], [p[1], c[1]])
            lines[i].set_3d_properties([p[2], c[2]])
            if caps is None:
                # Clean 패널: 주입 대상 체인만 초록으로 (어디가 바뀔지 미리 보여준다)
                lines[i].set_color('mediumseagreen' if child in highlight else 'lightgray')
                continue
            d = pen_by_bone.get(child, 0.0)
            bcol, ccol, alpha = mv.depth_style(d)
            if d <= 0.0:
                bcol = 'mediumseagreen' if child in highlight else 'dimgray'
                ccol = 'dodgerblue'
            lines[i].set_color(bcol)
            caps[i].set_data([p[0], c[0]], [p[1], c[1]])
            caps[i].set_3d_properties([p[2], c[2]])
            caps[i].set_color(ccol)
            caps[i].set_alpha(alpha)

    def pen_by_bone(pos_t):
        out = {}
        for (cap1, cap2, p1, c1, p2, c2), thr in zip(mv.TEST_PAIRS, pair_thr):
            dist = physics.capsule_distance(pos_t[BONE_MAP[p1]], pos_t[BONE_MAP[c1]],
                                            pos_t[BONE_MAP[p2]], pos_t[BONE_MAP[c2]]).item()
            pen = max(0.0, thr - dist)
            if pen > 0:
                for b in mv.CAPSULE_BONES[cap1] + mv.CAPSULE_BONES[cap2]:
                    out[b] = max(out.get(b, 0.0), pen)
        return out

    def update(k):
        pos_c = mv.get_pos_tensor(clean_py[k], offsets_py)
        pos_i = mv.get_pos_tensor(inj_py[k], offsets_py)
        draw_panel(lines_clean, None, pos_c.numpy(), {})
        draw_panel(lines_inj, caps_inj, pos_i.numpy(), pen_by_bone(pos_i))

        d = float(depths[k])
        inside = f0 <= k <= f1
        txt.set_text(f"frame {k}  |  {d:.2f} cm" + ("  [주입 구간]" if inside else ""))
        txt.set_color('red' if d > 1e-4 else 'dimgray')
        cursor.set_xdata([k, k])
        return lines_clean + lines_inj + caps_inj + [txt, cursor]

    ani = FuncAnimation(fig, update, frames=F, interval=1000 // max(FPS, 1), blit=False)
    # tight_layout은 위 GridSpec의 여백 설정(3D 패널을 크게 잡은 것)을 되돌리므로 쓰지 않는다.

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ani.save(out_path, writer='pillow', fps=FPS)
    print(f"  ✅ 저장: {out_path}")

    if show:
        _controls = mv.attach_viewer_controls(fig, ani, [ax1, ax2])
        plt.show()
    else:
        plt.close(fig)
    return out_path


def main(argv=None):
    if INJECT_MODE not in INJECT_MODES:
        raise ValueError(f"알 수 없는 INJECT_MODE: {INJECT_MODE!r} (지원: {list(INJECT_MODES)})")

    p = argparse.ArgumentParser(
        description="손상 주입(transient/persistent) 시각화 gif 생성. "
                    "인자를 생략하면 소스 상단 '소스 설정' 블록의 값을 쓴다.")
    p.add_argument("mode", nargs="?", default=None, choices=list(INJECT_MODES),
                   help="transient / persistent / both (기본: 소스 설정)")
    p.add_argument("--seed", type=int, default=INJECT_SEED, help="파일·주입 추첨 시드")
    p.add_argument("--out", default=OUT_DIR, help="gif 저장 폴더")
    p.add_argument("--file", default=TARGET_FILE, help="특정 .pt 고정")
    p.add_argument("--show", action="store_true", default=SHOW_WINDOW,
                   help="저장 후 창도 띄운다")
    args = p.parse_args(argv)

    mode = args.mode or INJECT_MODE
    if args.mode is None:
        print(f"ℹ️ 인자가 없어 소스 설정을 사용합니다 — INJECT_MODE='{mode}', "
              f"seed={args.seed} (바꾸려면 inject_viz.py 상단을 편집하세요).\n")

    # gif 저장 중에는 창을 띄우지 않는 백엔드가 안전하다 (--show 일 때만 GUI 유지)
    if not args.show:
        matplotlib.use("Agg", force=True)

    kinds = ('transient', 'persistent') if mode == 'both' else (mode,)
    made = []
    for kind in kinds:
        out = os.path.join(args.out, f"injection_{kind}.gif")
        made.append(make_injection_gif(kind, out, seed=args.seed,
                                       target_file=args.file, show=args.show))
    print("\n완료: " + ", ".join(made))
    if mode == 'both':
        print("두 gif의 아래 그래프를 비교하면 유형 차이가 분명하다 — "
              "transient는 봉우리 하나, persistent는 구간 내내 유지되는 고원.")


if __name__ == "__main__":
    main()
