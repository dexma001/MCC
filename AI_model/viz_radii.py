"""
캡슐 반지름(BONE_RADII) 동작 확인용 시각화 — "지금 물리 엔진이 보는 몸"을 눈으로 검증한다.

무엇을 보여주는가 (다섯 개의 서브커맨드 — 앞 넷은 정지 PNG, anim은 움직이는 애니메이션)
------------------------------------------------------------------------------
  table    : 반지름이 '어떻게 만들어지는가'. 해부학 비율 → 신장 스케일 → KAPPA 3층 파생,
             구판 대비 재배분, 학습 4페어의 임계값, 전신 112페어 임계값 분포.
  capsule  : 반지름이 '무엇이 되는가'. 실제 반지름으로 21개 캡슐을 3D 표면으로 렌더링한다.
             --mode both 면 legacy(구판) 옆에 나란히 그려 두께 재배분을 직접 비교한다.
  probe    : 판정이 '어떻게 이뤄지는가'. 한 페어를 확대해 두 캡슐 선분, Lumelsky 최근접점,
             축간거리 dist, 임계값 thr = r[c1]+r[c2], 침투 relu(thr-dist)를 한 화면에 그린다.
             최근접점에 반지름 r1/r2 구를 그리므로 '두 구의 겹침 = 침투'가 그대로 보인다.
  inject   : 손상이 '어떻게 채택되는가'. 실제 corruption.py 주입기를 돌린 뒤 프레임별 깊이,
             주입 구간, 목표 범위(1~4cm), coverage/중앙값과 채택 판정을 함께 그린다.
  anim     : 위를 '움직이는 형태'로. 기본은 matplotlib 창에서 실시간 재생이며(파일을 만들지
             않는다), gif 파일이 필요할 때만 --gif 를 준다. 뷰어 컨트롤은 viz_motion와 동일.
             --view probe(기본) 전신 캡슐 + 최근접점 확대 + 깊이 그래프 커서
             --view body        전신 캡슐 + 깊이 그래프
             --view both        신판/구판 전신을 나란히 — 같은 동작을 두 반지름이 어떻게
                                다르게 판정하는지 보여준다. 뼈대는 완전히 같다(FK는 반지름과
                                무관). 달라지는 건 살의 두께와 그 결과인 판정뿐이다.

⚠️ 좌표 규약 — 이 파일은 viz_motion의 Z-Up 어댑터를 쓰지 않는다.
   physics_module의 FK 결과(Unity Y-Up)를 축 치환(x, z, y)만 해서 그린다. 그래야 화면에
   그려진 기하와 물리 엔진이 판정한 기하가 '같은 좌표의 같은 점'임이 보장된다
   (어댑터를 거치면 그림이 맞아도 판정과 같은 수라는 보증이 사라진다).

⚠️ 반지름 전환은 전역을 건드리지 않는다.
   dataset_pipeline.RADII_MODE는 import 시점에 BONE_RADII를 확정하므로 런타임에 바꿔도
   이미 만들어진 dict에는 반영되지 않는다. 이 도구는 문서화된 방식대로
   make_bone_radii(kappa=...) / BONE_RADII_LEGACY 로 별도 dict를 만들어 쓴다
   (bone_radii_structure_and_usage.md §4.4).

실행 방법 (viz_motion / viz_inject 와 동일한 두 가지)
  (A) 소스 설정 + VS Code Run(Ctrl+F5) — 아래 '소스 설정' 블록을 고친다.
  (B) 명령줄 인자 (소스 설정보다 우선):
        python AI_model/viz_radii.py table
        python AI_model/viz_radii.py capsule --mode both
        python AI_model/viz_radii.py probe   --inject persistent      # --pair 0~3 고정 가능
        python AI_model/viz_radii.py inject  --inject persistent
        python AI_model/viz_radii.py anim    --view both --inject persistent   # 창에서 실시간 재생
        python AI_model/viz_radii.py anim    --view probe --fps 6              # 느리게 재생
        python AI_model/viz_radii.py anim    --view both --gif                 # gif로 저장할 때만
        python AI_model/viz_radii.py all --out claude_analysis/radii_figs --show
      공통 옵션: --kappa 0.4 (KAPPA만 바꿔 즉시 확인) / --seed / --file / --frame
      --show 는 '정지 그림(table/capsule/probe/inject)'을 창으로 띄우는 옵션이다.
      anim은 --show 와 무관하게 항상 창에서 실시간 재생한다 (스페이스바 일시정지 ·
      마우스 스크롤 확대 — viz_motion의 뷰어 컨트롤을 그대로 쓴다).
      --gif 를 주면 재생 대신 gif 파일로 저장한다 (--gif --show 면 저장 후 재생 창도 뜬다).

이 도구는 평가 CSV를 건드리지 않는다. evaluate.py에서는 전신 페어 정의(get_all_eval_pairs)만
가져오며, import는 __main__ 가드 때문에 아무것도 실행하지 않는다.
"""
import argparse
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import corruption
from dataset_pipeline import (ANATOMICAL_RADIUS_AT_REF, BONE_NAMES, BONE_RADII,
                              BONE_RADII_LEGACY, PARENTS, REF_STATURE_M, RADII_MODE,
                              SKELETON_STATURE_M, SOFT_TISSUE_KAPPA, get_split_files,
                              make_bone_radii)
from physics_module import DifferentiablePhysics
from train import COLLIDING_PAIRS

# 경로 (프로젝트 루트 실행이 표준, AI_model/ 안에서 실행해도 되도록 폴백 — viz_motion와 동일 규약)
PROCESSED_PT_DIR = "processed_motions_VMC" \
    if os.path.exists("processed_motions_VMC") else "../processed_motions_VMC"
OFFSET_CSV_PATH = "Sample_Data/Standard_BoneOffsets.csv" \
    if os.path.exists("Sample_Data/Standard_BoneOffsets.csv") \
    else "../Sample_Data/Standard_BoneOffsets.csv"

# =====================================================================
# 소스 설정 — 인자 없이 Run(Ctrl+F5)할 때 쓰이는 값
# =====================================================================
COMMAND = "capsule"            # "table" / "capsule" / "probe" / "inject" / "anim" / "all"
MODE = "both"              # capsule 전용: "anatomical" / "legacy" / "both"
KAPPA = None               # None이면 dataset_pipeline.SOFT_TISSUE_KAPPA. 예: 0.40
PAIR_INDEX = -1            # probe 전용: 학습 4페어 중 확대할 페어 (0~3). -1이면 가장 깊게 뚫린 페어
INJECT_KIND = "persistent"  # "persistent" / "transient" / "none"
SEED = 1234
TARGET_FILE = ""           # 특정 .pt 고정 (빈 문자열 = held-out 테스트셋에서 추첨)
FRAME = -1                 # probe 전용: -1이면 침투가 가장 깊은 프레임 자동 선택
SEQ_LEN = 30               # 학습 윈도우와 동일
ANIM_VIEW = "probe"        # anim 전용: "probe" / "body" / "both"
FPS = 10                   # 재생 속도 (느릴수록 관찰하기 쉽다 — viz_inject와 동일 기본값)
SAVE_GIF = False           # anim 전용: False = matplotlib 창에서 실시간 재생(기본), True = gif 저장
OUT_DIR = "claude_analysis/radii_figs"
SHOW_WINDOW = False        # 정지 그림(table/capsule/probe/inject)을 창으로 띄울지. anim과는 무관.

# 학습 4페어에 붙일 사람이 읽는 이름 (train.COLLIDING_PAIRS 순서와 1:1)
PAIR_LABELS = ["몸통 ↔ 왼팔(전완)", "몸통 ↔ 오른팔(전완)", "왼팔 ↔ 오른팔", "왼다리 ↔ 오른다리"]

_INJECTORS = {'transient': corruption.inject_transient,
              'persistent': corruption.inject_persistent}


def _use_korean_font():
    """한글 라벨이 두부로 깨지지 않게 폰트를 지정 (viz_inject와 동일 규약)."""
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Malgun Gothic", "NanumGothic", "Noto Sans KR", "AppleGothic", "Gulim"):
        if cand in available:
            plt.rcParams['font.family'] = cand
            plt.rcParams['axes.unicode_minus'] = False
            return cand
    print("⚠️ 한글 폰트를 찾지 못했습니다 — 라벨이 네모로 보일 수 있습니다.")
    return None


# =====================================================================
# 기하 유틸
# =====================================================================
def _plot_xyz(p):
    """Unity(Y-Up) 좌표 배열 [..., 3] → matplotlib 축 순서 (x, z, y). 축 치환만 한다."""
    p = np.asarray(p, dtype=float)
    return p[..., 0], p[..., 2], p[..., 1]


def closest_points(p1, q1, p2, q2):
    """
    선분 p1q1 과 p2q2 의 최근접점 쌍과 거리를 반환한다 (numpy, 단일 페어).

    physics_module.capsule_distance(Lumelsky)와 '같은 분기·같은 클램핑'을 따른다.
    호출 측이 physics 결과와 대조 검증하도록 거리도 함께 돌려준다 — 이 도구의 그림이
    실제 판정과 어긋나지 않음을 매 실행마다 확인하기 위한 장치다.
    """
    SMALL = 1e-8
    u, v, w = q1 - p1, q2 - p2, p1 - p2
    a, b, c = u @ u, u @ v, v @ v
    d, e = u @ w, v @ w
    D = a * c - b * b
    sD = tD = D

    if D < SMALL:                       # 두 선분이 거의 평행
        sN, tN, tD = 0.0, e, c
    else:
        sN, tN = b * e - c * d, a * e - b * d
        if sN < 0.0:
            sN, tN, tD = 0.0, e, c
        elif sN > sD:
            sN, tN, tD = sD, e + b, c

    if tN < 0.0:
        tN = 0.0
        sN, sD = min(max(-d, 0.0), a), a
    elif tN > tD:
        tN = tD
        sN, sD = min(max(-d + b, 0.0), a), a

    sc = 0.0 if abs(sN) < SMALL else sN / max(sD, SMALL)
    tc = 0.0 if abs(tN) < SMALL else tN / max(tD, SMALL)
    c1, c2 = p1 + sc * u, p2 + tc * v
    return c1, c2, float(np.linalg.norm(c1 - c2))


def _frame_basis(axis):
    """축 벡터에 수직인 정규직교 기저 (e1, e2)."""
    a = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, a)
    e1 /= (np.linalg.norm(e1) + 1e-12)
    return e1, np.cross(axis, e1)


def capsule_surfaces(p, q, r, n_theta=18, n_cap=6):
    """
    선분 p→q 를 반지름 r 캡슐(원기둥 + 반구 2개)의 표면 메시로 만든다.
    반환: [(X, Y, Z), ...]  — plot_surface에 그대로 넣을 수 있는 Unity 좌표 배열.
    캡슐은 '선분에서 거리 r 이내인 점의 집합'이므로, 이 표면이 곧 물리 엔진이 보는 몸이다.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = float(np.linalg.norm(q - p))
    th = np.linspace(0, 2 * np.pi, n_theta)
    if L < 1e-8:                                   # 길이 0 → 구 하나
        return [sphere_surface(p, r, n_theta, n_cap * 2)]

    u = (q - p) / L
    e1, e2 = _frame_basis(u)
    ring = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2      # [T, 3]

    s = np.linspace(0.0, 1.0, 2)[:, None, None]                     # 원기둥 옆면
    lat = p[None, None, :] + s * (q - p)[None, None, :] + r * ring[None, :, :]

    out = [(lat[..., 0], lat[..., 1], lat[..., 2])]
    for center, sign in ((q, 1.0), (p, -1.0)):                      # 양 끝 반구
        phi = np.linspace(0.0, np.pi / 2, n_cap)[:, None, None]
        cap = (center[None, None, :]
               + r * np.sin(phi) * (sign * u)[None, None, :]
               + r * np.cos(phi) * ring[None, :, :])
        out.append((cap[..., 0], cap[..., 1], cap[..., 2]))
    return out


def sphere_surface(center, r, n_theta=18, n_phi=12):
    """반지름 r 구 표면 (probe에서 최근접점의 '살'을 그릴 때 사용)."""
    th = np.linspace(0, 2 * np.pi, n_theta)
    ph = np.linspace(0, np.pi, n_phi)
    x = center[0] + r * np.outer(np.cos(th), np.sin(ph))
    y = center[1] + r * np.outer(np.sin(th), np.sin(ph))
    z = center[2] + r * np.outer(np.ones_like(th), np.cos(ph))
    return x, y, z


def _draw_surfaces(ax, surfs, color, alpha):
    for X, Y, Z in surfs:
        px, py, pz = _plot_xyz(np.stack([X, Y, Z], axis=-1))
        ax.plot_surface(px, py, pz, color=color, alpha=alpha,
                        linewidth=0, antialiased=False, shade=True)


def _equal_box(ax, pts, pad=0.12):
    """3D 축을 등방 스케일로 고정 — 이걸 안 하면 반지름이 눈에 왜곡되어 보인다."""
    px, py, pz = _plot_xyz(np.asarray(pts, float).reshape(-1, 3))
    ctr = np.array([px.mean(), py.mean(), pz.mean()])
    span = max(np.ptp(px), np.ptp(py), np.ptp(pz)) * (1 + pad) / 2 + 1e-3
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(ctr[2] - span, ctr[2] + span)
    ax.set_box_aspect((1, 1, 1))


# =====================================================================
# 데이터 / 물리 준비
# =====================================================================
def _ktag(kappa):
    """KAPPA를 명시적으로 바꿔 뽑은 그림은 파일명에 표시한다 — 기본 그림을 덮어쓰지 않도록."""
    return "" if kappa is None else f"_k{kappa}"


def _eval_pairs():
    """
    evaluate.py의 전신 평가 페어(위상거리 > 2)를 '그 정의 그대로' 가져온다.
    직접 다시 구현하면 evaluate와 조용히 어긋날 수 있어 정의를 공유한다.
    import 자체는 __main__ 가드 때문에 아무것도 실행하지 않고 CSV도 건드리지 않는다
    (measure_capsule_radii_calibration.py와 같은 규약).
    """
    from evaluate import get_all_eval_pairs
    return get_all_eval_pairs()


def radii_variants(kappa=None):
    """이름 → 반지름 dict. 전역 BONE_RADII를 변형하지 않는다 (§4.4의 권장 방식)."""
    k = SOFT_TISSUE_KAPPA if kappa is None else kappa
    anat = BONE_RADII if kappa is None else make_bone_radii(kappa=k)
    return {'anatomical': anat, 'legacy': BONE_RADII_LEGACY}, k


def load_window(seed=SEED, target_file="", seq_len=SEQ_LEN):
    """held-out 테스트셋에서 윈도우 하나를 뽑는다. 반환: (경로, [S, 87])"""
    if target_file:
        cands = [target_file]
    else:
        cands = get_split_files(PROCESSED_PT_DIR, split='test')
        random.Random(seed).shuffle(cands)
    for f in cands[:50]:
        full = torch.load(f)
        if full.shape[0] >= seq_len:
            return f, full[:seq_len].clone()
    raise RuntimeError(f"{seq_len}프레임 이상인 .pt를 찾지 못했습니다: {PROCESSED_PT_DIR}")


def fk_positions(physics, window):
    """[S, 87] → {본이름: [S, 3]} (Unity Y-Up, 물리 판정과 동일 좌표)."""
    return physics.forward_kinematics(window[:, :3], window[:, 3:])


def pair_depths_cm(physics, window, pairs):
    """[S, 87] → 페어별 프레임 침투 깊이 [S, P] (cm). evaluate와 같은 '선형' 깊이."""
    return physics.get_penetration_depths_from_quats(
        window[:, :3], window[:, 3:], pairs).detach().numpy() * 100.0


# =====================================================================
# 1) table — 반지름이 어떻게 만들어지는가
# =====================================================================
def fig_table(kappa=None, out_dir=OUT_DIR, show=False):
    variants, k = radii_variants(kappa)
    anat, legacy = variants['anatomical'], variants['legacy']
    scale = k * SKELETON_STATURE_M / REF_STATURE_M

    fig = plt.figure(figsize=(17, 10))
    fig.suptitle(f"BONE_RADII 신판 구조  |  KAPPA={k}  ·  신장 {SKELETON_STATURE_M} m  "
                 f"·  기준 {REF_STATURE_M} m  (현재 모듈 모드: {RADII_MODE})", fontsize=13)

    # (A) 3층 파생 — 고유 세그먼트 13개
    ax = fig.add_subplot(2, 2, 1)
    segs = list(ANATOMICAL_RADIUS_AT_REF.keys())
    a_ref = np.array([ANATOMICAL_RADIUS_AT_REF[s] for s in segs]) * 100
    a_sc = a_ref * (SKELETON_STATURE_M / REF_STATURE_M)
    a_fin = a_ref * scale
    y = np.arange(len(segs))
    ax.barh(y, a_ref, color='lightsteelblue', label=f"① 해부학 @{REF_STATURE_M}m")
    ax.barh(y, a_sc, color='cornflowerblue', label=f"② × 신장비 ({SKELETON_STATURE_M/REF_STATURE_M:.3f})")
    ax.barh(y, a_fin, color='navy', label=f"③ × KAPPA ({k}) = 최종")
    ax.set_yticks(y); ax.set_yticklabels(segs, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("반지름 (cm)"); ax.set_title("① 해부학 비율 → ② 신장 스케일 → ③ KAPPA", fontsize=11)
    ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)

    # (B) 신판 vs 구판 — 재배분
    ax = fig.add_subplot(2, 2, 2)
    order = sorted(BONE_NAMES, key=lambda b: -anat[b] / max(legacy[b], 1e-9))
    y = np.arange(len(order))
    ax.barh(y - 0.2, [legacy[b] * 100 for b in order], height=0.4, color='darkgray', label="구판(legacy)")
    ax.barh(y + 0.2, [anat[b] * 100 for b in order], height=0.4, color='seagreen', label="신판(anatomical)")
    for i, b in enumerate(order):
        ax.text(max(anat[b], legacy[b]) * 100 + 0.15, i, f"×{anat[b]/legacy[b]:.2f}", fontsize=7, va='center')
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("반지름 (cm)"); ax.set_title("신판 vs 구판 — '전부 두꺼워짐'이 아니라 재배분", fontsize=11)
    ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)

    # (C) 학습 4페어 임계값 = r[c1] + r[c2]
    ax = fig.add_subplot(2, 2, 3)
    names, r1s, r2s = [], [], []
    for i, ((p1, c1), (p2, c2)) in enumerate(COLLIDING_PAIRS):
        names.append(f"{PAIR_LABELS[i]}\n{p1}→{c1}  ×  {p2}→{c2}")
        r1s.append(anat[c1] * 100); r2s.append(anat[c2] * 100)
    y = np.arange(len(names))
    ax.barh(y, r1s, color='indianred', label="r[c1]")
    ax.barh(y, r2s, left=r1s, color='steelblue', label="r[c2]")
    for i, (a_, b_) in enumerate(zip(r1s, r2s)):
        ax.text(a_ + b_ + 0.2, i, f"thr = {a_+b_:.2f} cm", fontsize=9, va='center', fontweight='bold')
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("임계값 (cm)"); ax.set_xlim(0, max(np.array(r1s) + np.array(r2s)) * 1.35)
    ax.set_title("학습 4페어 임계값 — 캡슐 이름은 '부모→자식' 세그먼트", fontsize=11)
    ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)

    # (D) 전신 평가 페어(위상거리 > 2) 임계값 분포 — evaluate와 '같은' 페어 집합을 쓴다
    ax = fig.add_subplot(2, 2, 4)
    eval_pairs = _eval_pairs()
    thr_all = [(anat[c1] + anat[c2]) * 100 for (_, c1), (_, c2) in eval_pairs]
    thr_leg = [(legacy[c1] + legacy[c2]) * 100 for (_, c1), (_, c2) in eval_pairs]
    bins = np.linspace(0, max(max(thr_all), max(thr_leg)) * 1.05, 30)
    ax.hist(thr_leg, bins=bins, color='darkgray', alpha=0.7, label=f"구판 (중앙 {np.median(thr_leg):.1f}cm)")
    ax.hist(thr_all, bins=bins, color='seagreen', alpha=0.7, label=f"신판 (중앙 {np.median(thr_all):.1f}cm)")
    for i, ((_, c1), (_, c2)) in enumerate(COLLIDING_PAIRS):
        ax.axvline((anat[c1] + anat[c2]) * 100, color='crimson', lw=1.4, ls='--',
                   label="학습 4페어" if i == 0 else None)
    ax.set_xlabel("페어 임계값 r[c1]+r[c2] (cm)"); ax.set_ylabel("페어 수")
    ax.set_title(f"전신 평가 페어({len(eval_pairs)}개) 임계값 분포", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _finish(fig, f"radii_1_table{_ktag(kappa)}", out_dir, show)


# =====================================================================
# 2) capsule — 반지름이 무엇이 되는가 (3D 실물)
# =====================================================================
def fig_capsule(mode=MODE, kappa=None, seed=SEED, target_file="", frame=0,
                out_dir=OUT_DIR, show=False):
    variants, k = radii_variants(kappa)
    which = ['anatomical', 'legacy'] if mode == 'both' else [mode]
    fpath, window = load_window(seed, target_file)
    physics = DifferentiablePhysics(PARENTS, variants['anatomical'], offset_csv_path=OFFSET_CSV_PATH)
    gp = fk_positions(physics, window)
    f = max(0, min(frame, window.shape[0] - 1))
    P = {b: gp[b][f].detach().numpy() for b in BONE_NAMES}

    fig = plt.figure(figsize=(7.5 * len(which), 8.5))
    fig.suptitle(f"캡슐 = 선분(부모→자식) + 반지름 BONE_RADII[자식]   |   "
                 f"{os.path.basename(fpath)} frame {f}", fontsize=13)
    for i, name in enumerate(which):
        R_ = variants[name]
        ax = fig.add_subplot(1, len(which), i + 1, projection='3d')
        for b in BONE_NAMES:
            par = PARENTS[b]
            if par is None:
                continue
            _draw_surfaces(ax, capsule_surfaces(P[par], P[b], R_[b]),
                           'seagreen' if name == 'anatomical' else 'darkgray', 0.32)
            ax.plot(*_plot_xyz(np.stack([P[par], P[b]])), color='k', lw=1.0, alpha=0.55)
        pts = np.stack([P[b] for b in BONE_NAMES])
        ax.scatter(*_plot_xyz(pts), color='crimson', s=9, depthshade=False)
        _equal_box(ax, pts)
        tot = sum(R_[b] for b in BONE_NAMES if PARENTS[b]) * 100
        sub = f"KAPPA={k}" if name == 'anatomical' else "손튜닝 21상수"
        ax.set_title(f"{name}  ({sub})\n캡슐 반지름 합 {tot:.1f} cm", fontsize=11)
        ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_zlabel('y (up)')
        ax.view_init(elev=12, azim=-70)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig, f"radii_2_capsule_{mode}{_ktag(kappa)}", out_dir, show)


# =====================================================================
# 3) probe — 판정이 어떻게 이뤄지는가
# =====================================================================
def fig_probe(pair_index=PAIR_INDEX, kappa=None, seed=SEED, target_file="",
              inject_kind=INJECT_KIND, frame=FRAME, out_dir=OUT_DIR, show=False):
    variants, _ = radii_variants(kappa)
    R_ = variants['anatomical']
    physics = DifferentiablePhysics(PARENTS, R_, offset_csv_path=OFFSET_CSV_PATH)
    fpath, clean = load_window(seed, target_file)

    window, meta = clean, {'type': 'clean'}
    if inject_kind in _INJECTORS:
        window, meta = _INJECTORS[inject_kind](clean, physics, corruption.make_cfg(),
                                               random.Random(seed), COLLIDING_PAIRS)

    # pair_index < 0 이면 '실제로 가장 깊게 뚫린 페어'를 고른다. 주입은 본 하나만 돌리므로
    # 고정 페어를 보면 무충돌 화면이 나오기 쉽다 (예: 다리를 주입했는데 몸통↔팔을 보는 경우).
    depths_all = pair_depths_cm(physics, window, COLLIDING_PAIRS)      # [S, P]
    if pair_index < 0:
        pair_index = int(depths_all.max(axis=0).argmax())
    (p1n, c1n), (p2n, c2n) = COLLIDING_PAIRS[pair_index]
    depths = depths_all[:, pair_index]
    f = int(np.argmax(depths)) if frame < 0 else max(0, min(frame, window.shape[0] - 1))

    gp = fk_positions(physics, window)
    A1, B1 = gp[p1n][f].detach().numpy(), gp[c1n][f].detach().numpy()
    A2, B2 = gp[p2n][f].detach().numpy(), gp[c2n][f].detach().numpy()
    r1, r2 = R_[c1n], R_[c2n]
    thr = r1 + r2

    n1, n2, dist = closest_points(A1, B1, A2, B2)
    ref = float(physics.capsule_distance(gp[p1n][f], gp[c1n][f], gp[p2n][f], gp[c2n][f]))
    assert abs(dist - ref) < 1e-5, f"최근접점 계산이 physics와 어긋남: {dist} vs {ref}"
    pen = max(0.0, thr - dist)

    fig = plt.figure(figsize=(15, 7.5))
    fig.suptitle(f"충돌 판정 = relu( r[{c1n}] + r[{c2n}] - dist )   |   "
                 f"{PAIR_LABELS[pair_index]}   |   {os.path.basename(fpath)} · "
                 f"{meta.get('type')} · frame {f}", fontsize=13)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for (pa, pb, rr, col) in ((A1, B1, r1, 'indianred'), (A2, B2, r2, 'steelblue')):
        _draw_surfaces(ax, capsule_surfaces(pa, pb, rr), col, 0.22)
        ax.plot(*_plot_xyz(np.stack([pa, pb])), color=col, lw=2.5)
    # 최근접점의 '살' — 두 구가 겹치면 그 겹침이 곧 침투다
    _draw_surfaces(ax, [sphere_surface(n1, r1)], 'indianred', 0.5)
    _draw_surfaces(ax, [sphere_surface(n2, r2)], 'steelblue', 0.5)
    ax.plot(*_plot_xyz(np.stack([n1, n2])), color='k', lw=2.2, ls='--')
    ax.scatter(*_plot_xyz(np.stack([n1, n2])), color='k', s=28, depthshade=False)
    _equal_box(ax, np.stack([A1, B1, A2, B2, n1, n2]), pad=0.5)
    ax.set_title(f"{p1n}→{c1n} (r={r1*100:.2f}cm)  vs  {p2n}→{c2n} (r={r2*100:.2f}cm)", fontsize=10)
    ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_zlabel('y (up)')
    ax.view_init(elev=14, azim=-62)

    # 오른쪽: 1D 수직선 — dist vs thr, 그리고 프레임별 깊이
    ax = fig.add_subplot(2, 2, 2)
    ax.barh([0], [dist * 100], color='0.75', height=0.45, label="축간거리 dist")
    ax.barh([0], [thr * 100], color='none', edgecolor='crimson', lw=2.2, height=0.75,
            label="임계값 thr = r1+r2")
    if pen > 0:
        ax.barh([0], [pen * 100], left=[dist * 100], color='crimson', alpha=0.75,
                height=0.45, label="침투 relu(thr-dist)")
    ax.set_yticks([]); ax.set_xlabel("cm"); ax.legend(fontsize=8, loc='lower right')
    verdict = "충돌" if pen > 0 else "무충돌"
    ax.set_title(f"dist {dist*100:.2f} cm  vs  thr {thr*100:.2f} cm  →  "
                 f"침투 {pen*100:.2f} cm  ({verdict})", fontsize=11,
                 color='crimson' if pen > 0 else 'seagreen')
    ax.grid(axis='x', alpha=0.3)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(depths, color='crimson', lw=1.8)
    ax.axvline(f, color='k', ls=':', lw=1.2)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.fill_between(range(len(depths)), 0, depths, color='crimson', alpha=0.18)
    ax.set_xlabel("프레임"); ax.set_ylabel("침투 깊이 (cm)")
    ax.set_title(f"이 페어의 프레임별 깊이 (표시 프레임 {f})", fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, f"radii_3_probe_pair{pair_index}{_ktag(kappa)}", out_dir, show)


# =====================================================================
# 4) inject — 손상이 어떻게 채택되는가
# =====================================================================
def fig_inject(inject_kind=INJECT_KIND, kappa=None, seed=SEED, target_file="",
               out_dir=OUT_DIR, show=False):
    kind = inject_kind if inject_kind in _INJECTORS else 'persistent'
    variants, k = radii_variants(kappa)
    R_ = variants['anatomical']
    physics = DifferentiablePhysics(PARENTS, R_, offset_csv_path=OFFSET_CSV_PATH)
    cfg = corruption.make_cfg()
    fpath, clean = load_window(seed, target_file)
    injected, meta = _INJECTORS[kind](clean, physics, cfg, random.Random(seed), COLLIDING_PAIRS)

    d_clean = pair_depths_cm(physics, clean, COLLIDING_PAIRS)        # [S, 4]
    d_inj = pair_depths_cm(physics, injected, COLLIDING_PAIRS)
    fmax_clean, fmax_inj = d_clean.max(1), d_inj.max(1)
    S = len(fmax_inj)

    lo, hi = cfg['persistent_depth_range_cm']
    # ⚠️ meta['seg']는 구간이 아니라 '구간 종류' 문자열('full'/'onset'/'offset')이다.
    #    실제 프레임 구간은 meta['frames'] (클린 폴백 샘플에는 아예 없다).
    seg_kind = meta.get('seg', '-')
    s0, s1 = (int(meta['frames'][0]), int(meta['frames'][1])) if meta.get('frames') else (0, S - 1)
    inseg = fmax_inj[s0:s1 + 1]
    cov = float((inseg > 1e-4).mean()) if len(inseg) else 0.0
    med = float(np.sort(inseg)[(len(inseg) - 1) // 2]) if len(inseg) else 0.0

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(f"손상 주입 판정 — {kind}   |   {os.path.basename(fpath)}   |   "
                 f"주입 본 {meta.get('bone', '-')}  ·  KAPPA={k}", fontsize=13)

    # (A) 프레임별 최대 깊이 + 채택 기준
    ax = fig.add_subplot(2, 1, 1)
    ax.axvspan(s0, s1, color='gold', alpha=0.18, label=f"주입 구간 [{s0}, {s1}] ({seg_kind})")
    if kind == 'persistent':
        ax.axhspan(lo, hi, color='seagreen', alpha=0.13, label=f"깊이 목표 {lo}~{hi} cm (중앙값 기준)")
    ax.plot(fmax_clean, color='0.6', lw=1.6, ls='--', label="clean (주입 전)")
    ax.plot(fmax_inj, color='crimson', lw=2.0, label="injected (주입 후)")
    ax.axhline(0, color='k', lw=0.8)
    if kind == 'persistent':
        ax.axhline(med, color='darkgreen', lw=1.4, ls=':', label=f"구간 중앙값 {med:.2f} cm")
    ax.scatter(np.arange(s0, s1 + 1)[inseg <= 1e-4], np.zeros((inseg <= 1e-4).sum()),
               color='k', marker='x', s=40, zorder=5,
               label="구간 내 무관통 프레임 (coverage 감점)")
    ax.set_xlabel("프레임"); ax.set_ylabel("4페어 최대 침투 깊이 (cm)")
    ax.set_title("판정은 '한 프레임 max'가 아니라 '구간 coverage + 중앙값'으로 한다", fontsize=11)
    ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)

    # (B) 페어별 깊이 — 어느 페어가 뚫렸나
    ax = fig.add_subplot(2, 2, 3)
    for i in range(len(COLLIDING_PAIRS)):
        ax.plot(d_inj[:, i], lw=1.6, label=PAIR_LABELS[i])
    ax.axvspan(s0, s1, color='gold', alpha=0.18)
    ax.set_xlabel("프레임"); ax.set_ylabel("침투 깊이 (cm)")
    ax.set_title("학습 4페어 각각의 깊이 (주입은 4페어로만 판정한다)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (C) 판정 요약
    ax = fig.add_subplot(2, 2, 4); ax.axis('off')
    ok_cov = cov >= cfg['persistent_min_coverage']
    ok_med = lo <= med <= hi
    lines = [
        f"유형          : {meta.get('type')}"
        + ("   (fallback=클린 폴백)" if meta.get('fallback') else "")
        + ("   (out_of_range=차선 채택)" if meta.get('out_of_range') else ""),
        f"주입 본/축    : {meta.get('bone', '-')}  axis={meta.get('axis', '-')}",
        f"각도          : {meta.get('theta_deg', meta.get('theta_peak_deg', '-'))}°"
        f"   (라운드 {meta.get('tries', '-')})",
        "",
        f"임계값 자     : BONE_RADII (KAPPA={k}) → 4페어 thr = "
        + ", ".join(f"{(R_[c1]+R_[c2])*100:.1f}" for (_, c1), (_, c2) in COLLIDING_PAIRS) + " cm",
        # 그림 안에서는 이모지를 쓰지 않는다 — Malgun Gothic에 글리프가 없어 두부가 된다.
        f"구간 coverage : {cov*100:.1f} %   (기준 >= {cfg['persistent_min_coverage']*100:.0f} %)"
        f"  {'[OK]' if ok_cov else '[NG]'}",
        f"구간 중앙값   : {med:.2f} cm   (목표 {lo}~{hi} cm)  {'[OK]' if ok_med else '[NG]'}",
        f"구간 최대     : {inseg.max() if len(inseg) else 0:.2f} cm"
        f"   (meta max_depth_cm = {meta.get('max_depth_cm', '-')})",
        "",
        "[!] meta['collided']는 '충돌했는가'가 아니라",
        "   '주입기 내부 임계를 통과했는가'다 → 실제 판정은 위 깊이로 읽는다.",
        "",
        "반지름을 바꾸면 thr이 바뀌고, thr이 바뀌면 같은 각도라도",
        "채택되는 후보가 달라진다 → 학습 입력 자체가 달라진다.",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va='top', ha='left', fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig, f"radii_4_inject_{kind}{_ktag(kappa)}", out_dir, show)


# =====================================================================
# 5) anim — 위 셋을 '움직이는 형태'로 (viz_motion / viz_inject 규약의 gif)
# =====================================================================
def make_anim(view=ANIM_VIEW, inject_kind=INJECT_KIND, kappa=None, seed=SEED,
              target_file="", fps=FPS, out_dir=OUT_DIR, show=False, save_gif=SAVE_GIF):
    """
    한 윈도우(30프레임)를 재생하면서 캡슐과 판정이 어떻게 움직이는지 보여준다.

    기본 동작은 matplotlib 창에서의 '실시간 재생'이다 — 파일을 만들지 않는다.
    gif가 필요할 때만 save_gif=True(--gif)를 준다. 저장은 프레임을 전부 렌더링해야 해서
    느리므로, 반지름을 바꿔가며 눈으로 확인하는 평소 용도에는 창 재생이 맞다.
    (save_gif=True 이고 show=True 면 저장 후 재생 창도 띄운다.)

      view='probe' : [전신 캡슐] + [최근접점 확대] + 깊이 그래프   (기본)
      view='body'  : [전신 캡슐] + 깊이 그래프
      view='both'  : [신판 전신] + [구판 전신] + 깊이 그래프
                     ⚠️ 두 패널의 '뼈대'는 완전히 같다 — FK는 반지름과 무관하기 때문이다.
                        달라지는 것은 오직 살(캡슐 두께)과 그로부터 나오는 판정이다.
                        같은 동작에서 신판만 빨개지는 프레임이 곧 '구판이 놓치던 클리핑'이다.

    색 규칙은 viz_motion.depth_style을 그대로 쓴다 — 같은 깊이가 도구마다 다른 색으로
    보이면 안 되므로 색 정의는 한 곳에만 둔다.
    """
    from matplotlib.animation import FuncAnimation
    import viz_motion as mv          # depth_style / 뷰어 컨트롤만 쓴다 (지연 import)

    variants, k = radii_variants(kappa)
    R_anat, R_leg = variants['anatomical'], variants['legacy']
    physics = DifferentiablePhysics(PARENTS, R_anat, offset_csv_path=OFFSET_CSV_PATH)
    physics_leg = DifferentiablePhysics(PARENTS, R_leg, offset_csv_path=OFFSET_CSV_PATH)

    fpath, clean = load_window(seed, target_file)
    window, meta = clean, {'type': 'clean'}
    if inject_kind in _INJECTORS:
        window, meta = _INJECTORS[inject_kind](clean, physics, corruption.make_cfg(),
                                               random.Random(seed), COLLIDING_PAIRS)

    S = window.shape[0]
    gp = fk_positions(physics, window)
    POS = {b: gp[b].detach().numpy() for b in BONE_NAMES}          # {본: [S, 3]}
    d_anat = pair_depths_cm(physics, window, COLLIDING_PAIRS)      # [S, 4] cm
    d_leg = pair_depths_cm(physics_leg, window, COLLIDING_PAIRS)

    pair_i = int(d_anat.max(axis=0).argmax())                      # 확대할 페어는 통째로 고정
    (p1n, c1n), (p2n, c2n) = COLLIDING_PAIRS[pair_i]
    r1, r2 = R_anat[c1n], R_anat[c2n]
    thr = r1 + r2

    def pen_by_bone(frame_i, depths_cm):
        """페어별 깊이(cm) → 본별 깊이(m). 캡슐 구간에 속한 본에만 배분한다."""
        out = {}
        for j, (cap1, cap2, *_) in enumerate(mv.TEST_PAIRS):
            d_m = float(depths_cm[frame_i, j]) / 100.0
            if d_m > 0:
                for b in mv.CAPSULE_BONES[cap1] + mv.CAPSULE_BONES[cap2]:
                    out[b] = max(out.get(b, 0.0), d_m)
        return out

    # ---- 레이아웃 (viz_inject와 같은 철학: 3D를 크게, 그래프는 조역) ----
    n3d = 1 if view == 'body' else 2
    fig = plt.figure(figsize=(7.6 * n3d, 8.6))
    fig.suptitle(f"BONE_RADII 동작 확인 — {view}  |  {os.path.basename(fpath)}  |  "
                 f"{meta.get('type')}"
                 + (f" · 주입 본 {meta['bone']}" if meta.get('bone') else "")
                 + f"  |  KAPPA={k}", fontsize=13, fontweight='bold')
    gs = fig.add_gridspec(2, n3d, height_ratios=[3.4, 1.0],
                          left=0.04, right=0.98, top=0.92, bottom=0.08,
                          wspace=0.03, hspace=0.14)
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    ax_b = fig.add_subplot(gs[0, 1], projection='3d') if n3d == 2 else None
    axg = fig.add_subplot(gs[1, :])

    # 축 범위는 전 프레임을 감싸도록 '한 번만' 정한다 (프레임마다 다시 잡으면 화면이 요동친다)
    allpos = np.stack([POS[b] for b in BONE_NAMES], axis=1).reshape(-1, 3)
    bx, by, bz = _plot_xyz(allpos)
    ctr = np.array([bx.mean(), by.mean(), bz.mean()])
    span = max(np.ptp(bx), np.ptp(by), np.ptp(bz)) * 0.56 + 0.02

    seg_pts = np.concatenate([POS[n][:, None, :] for n in (p1n, c1n, p2n, c2n)], axis=1)
    zx, zy, zz = _plot_xyz(seg_pts.reshape(-1, 3))
    zctr = np.array([zx.mean(), zy.mean(), zz.mean()])
    zspan = max(np.ptp(zx), np.ptp(zy), np.ptp(zz)) * 0.62 + max(r1, r2) * 1.5

    def _reset(ax, center, half, title, elev=12, azim=-70, zoom=1.25):
        ax.clear()
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        # 3D 축 기본 여백이 커서 뼈대가 실제보다 작아 보인다 — viz_inject와 같은 보정.
        try:
            ax.set_box_aspect((1, 1, 1), zoom=zoom)
        except TypeError:                      # 구버전 matplotlib에는 zoom 인자가 없다
            ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(labelsize=6, pad=0)
        ax.set_title(title, fontsize=10)

    def draw_body(ax, radii, depths_cm, f, label):
        _reset(ax, ctr, span, label)
        pen = pen_by_bone(f, depths_cm)
        for b in BONE_NAMES:
            par = PARENTS[b]
            if par is None:
                continue
            _, col, alpha = mv.depth_style(pen.get(b, 0.0))
            _draw_surfaces(ax, capsule_surfaces(POS[par][f], POS[b][f], radii[b],
                                                n_theta=12, n_cap=4), col, alpha)
            ax.plot(*_plot_xyz(np.stack([POS[par][f], POS[b][f]])), color='k', lw=1.0, alpha=0.5)
        hit = float(depths_cm[f].max())
        ax.text2D(0.02, 0.97, f"frame {f}   최대 침투 {hit:.2f} cm", transform=ax.transAxes,
                  fontsize=10, va='top', fontweight='bold',
                  color='crimson' if hit > 1e-4 else 'dimgray')

    def draw_probe(ax, f):
        A1, B1 = POS[p1n][f], POS[c1n][f]
        A2, B2 = POS[p2n][f], POS[c2n][f]
        n1, n2, dist = closest_points(A1, B1, A2, B2)
        pen = max(0.0, thr - dist)
        _reset(ax, zctr, zspan,
               f"{p1n}→{c1n} (r={r1*100:.2f}cm)  vs  {p2n}→{c2n} (r={r2*100:.2f}cm)",
               zoom=1.15)
        for (pa, pb, rr, col) in ((A1, B1, r1, 'indianred'), (A2, B2, r2, 'steelblue')):
            _draw_surfaces(ax, capsule_surfaces(pa, pb, rr, n_theta=14, n_cap=5), col, 0.20)
            ax.plot(*_plot_xyz(np.stack([pa, pb])), color=col, lw=2.5)
        _draw_surfaces(ax, [sphere_surface(n1, r1, 14, 10)], 'indianred', 0.5)
        _draw_surfaces(ax, [sphere_surface(n2, r2, 14, 10)], 'steelblue', 0.5)
        ax.plot(*_plot_xyz(np.stack([n1, n2])), color='k', lw=2.0, ls='--')
        ax.text2D(0.02, 0.97,
                  f"dist {dist*100:.2f}  vs  thr {thr*100:.2f}  ->  침투 {pen*100:.2f} cm",
                  transform=ax.transAxes, fontsize=10, va='top', fontweight='bold',
                  color='crimson' if pen > 0 else 'seagreen')

    # ---- 깊이 그래프 (정적 부분은 한 번만 그린다) ----
    fmax_a, fmax_l = d_anat.max(1), d_leg.max(1)
    f0, f1 = (int(meta['frames'][0]), int(meta['frames'][1])) if meta.get('frames') else (0, S - 1)
    axg.axvspan(f0 - 0.5, f1 + 0.5, color='orange', alpha=0.15,
                label=f"주입 구간 ({f0}~{f1}{', ' + meta['seg'] if meta.get('seg') else ''})")
    axg.axhspan(1.0, 4.0, color='green', alpha=0.08, label="목표 1~4cm")
    axg.plot(range(S), fmax_a, '-o', ms=3, lw=1.8, color='crimson', label="신판(anatomical) 판정")
    axg.plot(range(S), fmax_l, '--', lw=1.5, color='gray', label="구판(legacy) 판정")
    cursor = axg.axvline(0, color='black', lw=1.8)
    axg.set_xlim(-0.5, S - 0.5)
    axg.set_ylim(0, max(4.5, float(max(fmax_a.max(), fmax_l.max())) * 1.15))
    axg.set_xlabel("frame", fontsize=8, labelpad=1)
    axg.set_ylabel("penetration (cm)", fontsize=8, labelpad=2)
    axg.tick_params(labelsize=7, pad=1)
    axg.grid(alpha=0.3)
    axg.legend(loc='upper right', fontsize=7, ncol=4, framealpha=0.85,
               borderpad=0.3, handlelength=1.4, columnspacing=1.0)

    def update(f):
        if view == 'both':
            draw_body(ax_a, R_anat, d_anat, f, f"신판 anatomical (KAPPA={k})")
            draw_body(ax_b, R_leg, d_leg, f, "구판 legacy (손튜닝 21상수)")
        else:
            draw_body(ax_a, R_anat, d_anat, f, f"전신 캡슐 — anatomical (KAPPA={k})")
            if ax_b is not None:
                draw_probe(ax_b, f)
        cursor.set_xdata([f, f])
        return []

    ani = FuncAnimation(fig, update, frames=S, interval=1000 // max(fps, 1), blit=False)

    path = None
    if save_gif:
        os.makedirs(out_dir, exist_ok=True)
        kind_tag = inject_kind if inject_kind in _INJECTORS else 'clean'
        path = os.path.join(out_dir, f"radii_5_anim_{view}_{kind_tag}{_ktag(kappa)}.gif")
        ani.save(path, writer='pillow', fps=fps)
        # 이모지를 쓰지 않는다 — cp949 콘솔(윈도우 기본)에서 UnicodeEncodeError로 죽는다.
        print(f"[saved] {path}")
        if not show:
            plt.close(fig)
            return path

    # 실시간 재생. ani/controls를 지역 변수로 붙들고 있어야 창이 뜬 동안 GC되지 않는다
    # (FuncAnimation은 참조가 끊기면 타이머가 죽어 정지 화면이 된다).
    if matplotlib.get_backend().lower() == "agg":
        print("[!] 현재 matplotlib 백엔드가 Agg(비대화형)라 창이 뜨지 않습니다 — "
              "gif가 필요하면 --gif 를 주세요.")
    _controls = mv.attach_viewer_controls(fig, ani, [a for a in (ax_a, ax_b) if a])
    print(f"[play] {view} · {S}프레임 · {fps} fps  (스페이스바 일시정지 · 스크롤 확대)")
    plt.show()
    del _controls
    return path


# =====================================================================
# 실행부
# =====================================================================
def _finish(fig, name, out_dir, show):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=130)
    # 이모지를 쓰지 않는다 — cp949 콘솔(윈도우 기본)에서 UnicodeEncodeError로 죽는다.
    print(f"[saved] {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="BONE_RADII 캡슐 구성 / 충돌·손상 판정 시각화")
    ap.add_argument("command", nargs="?", default=COMMAND,
                    choices=["table", "capsule", "probe", "inject", "anim", "all"])
    ap.add_argument("--view", default=ANIM_VIEW, choices=["probe", "body", "both"],
                    help="anim 전용 화면 구성 (기본 probe)")
    ap.add_argument("--fps", type=int, default=FPS, help="anim 재생 속도")
    ap.add_argument("--gif", action="store_true", default=SAVE_GIF,
                    help="anim을 창에서 재생하는 대신 gif 파일로 저장한다 (기본: 창 재생)")
    ap.add_argument("--mode", default=MODE, choices=["anatomical", "legacy", "both"])
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--pair", type=int, default=PAIR_INDEX,
                    choices=range(-1, len(COLLIDING_PAIRS)),
                    help="확대할 학습 페어 (0~3). -1 = 가장 깊게 뚫린 페어 자동 선택")
    ap.add_argument("--inject", default=INJECT_KIND, choices=["persistent", "transient", "none"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--file", default=TARGET_FILE)
    ap.add_argument("--frame", type=int, default=FRAME)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--show", action="store_true", default=SHOW_WINDOW,
                    help="정지 그림을 창으로 띄운다 (anim은 --show 없이도 창에서 재생한다)")
    a = ap.parse_args(argv)

    # anim은 기본이 '창에서 실시간 재생'이므로 대화형 백엔드가 필요하다.
    # gif 저장만 할 때는 창을 띄울 이유가 없어 예전처럼 Agg로 둔다.
    live_anim = a.command in ("anim", "all") and not a.gif
    if not a.show and not live_anim:
        matplotlib.use("Agg")
    _use_korean_font()

    cmd = a.command
    if cmd in ("table", "all"):
        fig_table(a.kappa, a.out, a.show)
    if cmd in ("capsule", "all"):
        fig_capsule(a.mode, a.kappa, a.seed, a.file, max(a.frame, 0), a.out, a.show)
    if cmd in ("probe", "all"):
        fig_probe(a.pair, a.kappa, a.seed, a.file, a.inject, a.frame, a.out, a.show)
    if cmd in ("inject", "all"):
        fig_inject(a.inject if a.inject != "none" else "persistent",
                   a.kappa, a.seed, a.file, a.out, a.show)
    if cmd in ("anim", "all"):
        make_anim(a.view, a.inject, a.kappa, a.seed, a.file, a.fps, a.out, a.show, a.gif)


if __name__ == "__main__":
    main()
