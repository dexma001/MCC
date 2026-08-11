import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from scipy.spatial.transform import Rotation as R
import itertools

# ==========================================
# ⚙️ 1. 시스템 하이퍼파라미터 및 예외 룰
# ==========================================
# 아담한 체형 스케일에 맞춘 캡슐 반경 축소 (단위: 미터)
# ⚠️ 이 테이블은 dataset_pipeline.BONE_RADII 와 '이름은 같지만 값이 다르다'
#    (여기: Hips 0.08 / Spine 0.12 / Chest 0.10  vs  파이프라인: 0.06 / 0.04 / 0.08).
#    이 스크립트는 학습/평가 파이프라인과 무관한 독립 시각화용이므로 별도 상수로 둔다.
#    두 값을 통일하면 '동작 변경'이 되므로 임의로 맞추지 말 것.
BONE_RADII = {
    'Hips': 0.08, 'Spine': 0.12, 'Chest': 0.10, 'Neck': 0.03, 'Head': 0.08,
    'LeftShoulder': 0.03, 'LeftUpperArm': 0.03, 'LeftLowerArm': 0.02, 'LeftHand': 0.02,
    'RightShoulder': 0.03, 'RightUpperArm': 0.03, 'RightLowerArm': 0.02, 'RightHand': 0.02,
    'LeftUpperLeg': 0.05, 'LeftLowerLeg': 0.04, 'LeftFoot': 0.03, 'LeftToes': 0.02,
    'RightUpperLeg': 0.05, 'RightLowerLeg': 0.04, 'RightFoot': 0.03, 'RightToes': 0.02
}

# 양 다리 간의 억울한 충돌(차렷 자세 등) 무시 토글
IGNORE_LEG_TO_LEG_COLLISION = True
LEG_BONES = {'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot', 'LeftToes',
             'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'RightToes'}

# 데이터 로드 (원하시는 파일명으로 변경하여 사용하세요)
csv_file = r"Sample_Data/Bandai_Dataset_csv_modi_tot/Bandai_Dataset_2_csv/dataset-2_walk-turn-right_exhausted_027.csv"
df = pd.read_csv(csv_file)
frames = sorted(df['Frame'].unique())

# 뼈대 계층 구조
parents = {
    'Hips': None,
    'Spine': 'Hips', 'Chest': 'Spine', 'Neck': 'Chest', 'Head': 'Neck',
    'LeftShoulder': 'Chest', 'LeftUpperArm': 'LeftShoulder', 'LeftLowerArm': 'LeftUpperArm', 'LeftHand': 'LeftLowerArm',
    'RightShoulder': 'Chest', 'RightUpperArm': 'RightShoulder', 'RightLowerArm': 'RightUpperArm', 'RightHand': 'RightLowerArm',
    'LeftUpperLeg': 'Hips', 'LeftLowerLeg': 'LeftUpperLeg', 'LeftFoot': 'LeftLowerLeg', 'LeftToes': 'LeftFoot',
    'RightUpperLeg': 'Hips', 'RightLowerLeg': 'RightUpperLeg', 'RightFoot': 'RightLowerLeg', 'RightToes': 'RightFoot'
}
valid_bones = df['BoneName'].unique()
bones_to_draw = [(parents[b], b) for b in valid_bones if parents.get(b) in valid_bones]

# ==========================================
# 🧠 2. 위상 수학적 거리 캐싱 (촌수 필터링용)
# ==========================================
adj_list = {b: [] for b in valid_bones}
for c, p in parents.items():
    if c in valid_bones and p in valid_bones:
        adj_list[p].append(c)
        adj_list[c].append(p)

def get_topological_distance(start, target):
    if start == target: return 0
    q = [(start, 0)]
    visited = {start}
    while q:
        curr, dist = q.pop(0)
        if curr == target: return dist
        for nxt in adj_list.get(curr, []):
            if nxt not in visited:
                visited.add(nxt); q.append((nxt, dist + 1))
    return 999

topo_dist_cache = {(b1, b2): get_topological_distance(b1, b2) for b1 in valid_bones for b2 in valid_bones}

# ==========================================
# 🌟 3. 자동 영점 정렬 알고리즘 (Auto-Calibration)
# ==========================================
frame0_data = df[df['Frame'] == frames[0]].set_index('BoneName')
raw_pos, raw_rot = {}, {}

for bone in parents.keys():
    if bone not in frame0_data.index: continue
    row = frame0_data.loc[bone]
    local_p = np.array([row['px'], -row['pz'], -row['py']])
    lr = R.from_quat([-row['qx'], -row['qz'], -row['qy'], row['qw']])
    if bone != 'Hips': local_p *= 100.0  
        
    p_bone = parents[bone]
    if p_bone is None or p_bone not in raw_pos:
        raw_pos[bone] = local_p; raw_rot[bone] = lr
    else:
        raw_pos[bone] = raw_pos[p_bone] + raw_rot[p_bone].apply(local_p)
        raw_rot[bone] = raw_rot[p_bone] * lr

# Up 벡터와 Right 벡터 추출하여 로컬 축 생성
char_up = raw_pos['Neck'] - raw_pos['Hips']
char_up /= np.linalg.norm(char_up)
char_right = raw_pos['RightUpperLeg'] - raw_pos['LeftUpperLeg']
char_right /= np.linalg.norm(char_right)
char_forward = np.cross(char_up, char_right)
char_forward /= np.linalg.norm(char_forward)
char_right = np.cross(char_forward, char_up) # 직교성 보정

M_char = np.column_stack((char_right, char_forward, char_up))
root_correction = R.from_matrix(M_char.T)

# ==========================================
# 📏 4. 수학 코어: 두 선분 간의 최단 거리 연산
# ==========================================
def closest_distance_between_lines(P1, Q1, P2, Q2):
    u = Q1 - P1; v = Q2 - P2; w = P1 - P2
    a = np.dot(u, u); b = np.dot(u, v); c = np.dot(v, v)
    d = np.dot(u, w); e = np.dot(v, w); D = a*c - b*b
    sD = D; tD = D
    
    if D < 1e-8:
        sN, sD, tN, tD = 0.0, 1.0, e, c
    else:
        sN, tN = (b*e - c*d), (a*e - b*d)
        if sN < 0.0: sN, tN, tD = 0.0, e, c
        elif sN > sD: sN, tN, tD = sD, e + b, c
            
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0: sN = 0.0
        elif -d > a: sN = sD
        else: sN, sD = -d, a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0: sN = 0.0
        elif (-d + b) > a: sN = sD
        else: sN, sD = (-d + b), a
            
    sc = 0.0 if np.abs(sN) < 1e-8 else sN / sD
    tc = 0.0 if np.abs(tN) < 1e-8 else tN / tD
    
    dP = w + (sc * u) - (tc * v)
    return np.linalg.norm(dP)

# ==========================================
# 🏃 5. FK 연산 및 정밀 캡슐 충돌 검사 루프
# ==========================================
all_positions = []
all_collisions_per_frame = []

for f in frames:
    frame_data = df[df['Frame'] == f].set_index('BoneName')
    global_pos = {}
    global_rot = {}
    
    for bone in parents.keys():
        if bone not in frame_data.index: continue
        row = frame_data.loc[bone]
        local_p = np.array([row['px'], -row['pz'], -row['py']])
        lr = R.from_quat([-row['qx'], -row['qz'], -row['qy'], row['qw']])
        if bone != 'Hips': local_p *= 100.0  
            
        p_bone = parents[bone]
        if p_bone is None or p_bone not in global_pos:
            # 💡 [핵심] Hips에 오토 캘리브레이션 쿼터니언을 곱하여 항상 똑바로 세움
            global_rot[bone] = root_correction * lr
            global_pos[bone] = root_correction.apply(local_p)
        else:
            global_pos[bone] = global_pos[p_bone] + global_rot[p_bone].apply(local_p)
            global_rot[bone] = global_rot[p_bone] * lr
            
    all_positions.append(global_pos)
    
    collisions_in_this_frame = []
    line_segments = [(p, c) for p, c in bones_to_draw if p in global_pos and c in global_pos]

    for (p1, c1), (p2, c2) in itertools.combinations(line_segments, 2):
        if c1 not in BONE_RADII or c2 not in BONE_RADII: continue
        
        # [필터링 1] 트리의 촌수가 2 이하인 조부모/형제 관절 면제
        min_topo_dist = min(
            topo_dist_cache[(p1, p2)], topo_dist_cache[(p1, c2)],
            topo_dist_cache[(c1, p2)], topo_dist_cache[(c1, c2)]
        )
        if min_topo_dist <= 2: continue

        # [필터링 2] 양 다리 간의 충돌 무시 (차렷 자세 등)
        # 기본 판정 기준: 두 캡슐 끝점(자식 뼈) 반지름의 합
        threshold = BONE_RADII[c1] + BONE_RADII[c2]
        if IGNORE_LEG_TO_LEG_COLLISION:
            is_leg1 = p1 in LEG_BONES or c1 in LEG_BONES
            is_leg2 = p2 in LEG_BONES or c2 in LEG_BONES
            if is_leg1 and is_leg2:
                # 다리끼리의 검사일 경우, 충돌 허용치를 넓혀줍니다.
                # 캡슐의 두께(Radius)를 40%로 대폭 깎아서 뼈대 중심끼리 정말 가까워졌을 때만 충돌로 판정합니다.
                threshold *= 0.4

        dist = closest_distance_between_lines(global_pos[p1], global_pos[c1], global_pos[p2], global_pos[c2])

        if dist < threshold:
            collisions_in_this_frame.append(((p1, c1), (p2, c2)))
            
    all_collisions_per_frame.append(collisions_in_this_frame)

# ==========================================
# 🎥 6. Matplotlib 3D 렌더링 및 UI
# ==========================================
fig = plt.figure(figsize=(10, 10))
plt.subplots_adjust(bottom=0.2) 
ax = fig.add_subplot(111, projection='3d')
lines = [ax.plot([], [], [], 'o-', lw=3, markersize=4, color='lightgray')[0] for _ in bones_to_draw]
dist_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, color='black', fontsize=11, va='top')

def update(frame_idx):
    pos = all_positions[frame_idx]
    current_collisions = all_collisions_per_frame[frame_idx]
    
    if len(current_collisions) > 0:
        status_text = f"🚨 CAPSULE COLLISION ({len(current_collisions)} pairs)\n"
        for (seg1, seg2) in current_collisions[:3]: 
            status_text += f" - [{seg1[0]}~{seg1[1]}] & [{seg2[0]}~{seg2[1]}]\n"
        dist_text.set_color('red')
    else:
        status_text = "✅ SAFE (No Collisions)"
        dist_text.set_color('green')
    dist_text.set_text(f"Frame: {frames[frame_idx]}\n{status_text}")

    colliding_segments = set()
    for (seg1, seg2) in current_collisions:
        colliding_segments.add(seg1); colliding_segments.add(seg2)

    for i, (p, c) in enumerate(bones_to_draw):
        if p in pos and c in pos:
            lines[i].set_data([pos[p][0], pos[c][0]], [pos[p][1], pos[c][1]])
            lines[i].set_3d_properties([pos[p][2], pos[c][2]])
            
            if (p, c) in colliding_segments:
                lines[i].set_color('red'); lines[i].set_linewidth(6.0)
            else:
                lines[i].set_color('dimgray' if len(current_collisions) == 0 else 'lightgray')
                lines[i].set_linewidth(2.5)

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 2)
    # 정면에서 바라보도록 카메라 각도 고정 (오토 캘리브레이션 덕분에 항상 정면을 봅니다)
    ax.view_init(elev=20, azim=45) 
    return lines + [dist_text]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=33, blit=False)

# [UI 컨트롤러]
ax_play = plt.axes([0.15, 0.05, 0.15, 0.05])
btn_play = Button(ax_play, 'Pause', hovercolor='0.9')
is_playing = True
def toggle_play(event):
    global is_playing
    if is_playing: ani.pause(); btn_play.label.set_text('Play')
    else: ani.resume(); btn_play.label.set_text('Pause')
    is_playing = not is_playing
    fig.canvas.draw_idle()
btn_play.on_clicked(toggle_play)

ax_speed = plt.axes([0.45, 0.05, 0.4, 0.05])
slider_speed = Slider(ax_speed, 'Speed', 0.1, 3.0, valinit=1.0, valfmt='%0.1f x')
def update_speed(val): ani.event_source.interval = int(33 / val)
slider_speed.on_changed(update_speed)

plt.show()