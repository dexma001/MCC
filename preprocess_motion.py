import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def is_valid_motion(df, file_name):
    """
    조건을 만족하지 않는 훼손된 파일을 걸러내어 폐기(버림) 처리합니다.
    """
    req_cols = ['Frame', 'BoneName', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw']
    
    if df.empty: 
        return False, "빈 파일"
    if not all(col in df.columns for col in req_cols): 
        return False, "필수 컬럼 누락"
    if df.isnull().values.any(): 
        return False, "결측치(NaN) 포함"
    if 'Hips' not in df['BoneName'].values: 
        return False, "Hips(루트 본) 누락"
    if 0 not in df['Frame'].values: 
        return False, "Frame 0 (시작 프레임) 누락"
        
    return True, "Valid"

def get_upaxis_correction(hips_quats, file_name=""):
    """
    Hips 회전들로부터 몸통의 up 방향을 복원하여 파일의 up-axis를 판별하고,
    이를 Unity Y-Up(+Y)으로 세우는 글로벌 보정 회전(scipy Rotation)을 반환합니다.
    이미 +Y-Up이면 None을 반환합니다.

    - up 벡터 = Hips 회전을 로컬 +Y(척추 방향)에 적용한 값 (전 프레임 평균)
      → 위치/이동과 무관하므로 Z-Up/Y-Up 혼재 데이터를 안정적으로 구분합니다.
    """
    up = R.from_quat(hips_quats).apply([0.0, 1.0, 0.0]).mean(axis=0)
    axis = int(np.argmax(np.abs(up)))   # 0:X, 1:Y, 2:Z
    sign = 1.0 if up[axis] >= 0 else -1.0

    if axis == 1:
        # 이미 Y축이 up. +Y면 보정 불필요, -Y면 뒤집힘.
        if sign > 0:
            return None
        print(f"  -> [보정] 거꾸로 뒤집힌 모션(-Y) 일으켜 세움: {file_name}")
        return R.from_euler('x', 180, degrees=True)

    if axis == 2:
        # Z-Up: +Z는 -90°, -Z는 +90° X축 회전으로 +Y 정렬
        angle = -90 if sign > 0 else 90
        print(f"  -> [보정] Z축 Up 모션 일으켜 세움({'+' if sign>0 else '-'}Z): {file_name}")
        return R.from_euler('x', angle, degrees=True)

    # axis == 0, X-Up: +X는 +90°, -X는 -90° Z축 회전으로 +Y 정렬
    angle = 90 if sign > 0 else -90
    print(f"  -> [보정] X축 Up 모션 일으켜 세움({'+' if sign>0 else '-'}X): {file_name}")
    return R.from_euler('z', angle, degrees=True)


def normalize_motion_file(file_path, output_path):
    try:
        df = pd.read_csv(file_path)
        
        # 0. 무결성 검사 (불합격 시 False 반환하여 파일 저장 안 함)
        is_valid, reason = is_valid_motion(df, file_path.name)
        if not is_valid:
            return False, f"폐기됨: {reason}"
            
        q_cols = ['qx', 'qy', 'qz', 'qw']
        pos_cols = ['px', 'py', 'pz']
        hips_mask = df['BoneName'] == 'Hips'
        
        # 1. 쿼터니언 실수부(qw) 양수화
        neg_qw_mask = df['qw'] < 0
        df.loc[neg_qw_mask, q_cols] *= -1.0

        # ----------------------------------------------------
        # 2. 방향(Orientation) 기반 Up-Axis 탐지 및 일으켜 세우기
        # ----------------------------------------------------
        # 데이터셋에는 Y-Up 파일과 Z-Up 파일이 섞여 있습니다.
        # 루트 본(Hips)의 월드 '위치'는 이동(locomotion)과 월드 오프셋 때문에
        # 축 판별에 쓸 수 없습니다. 대신 Hips의 '회전'으로부터 몸통의 up 방향을
        # 복원하여 판별합니다. up 방향 = Hips 회전을 로컬 +Y(척추 방향)에 적용한 벡터이며,
        # 이동/위치와 무관하므로 신뢰할 수 있습니다. 전 프레임 평균으로 노이즈를 줄입니다.
        r_corr = get_upaxis_correction(df.loc[hips_mask, q_cols].values, file_path.name)

        # 회전 보정 행렬(r_corr)이 생성되었다면(누워있었다면) 전체 Hips 데이터에 적용
        if r_corr is not None:
            # Hips 좌표 회전 적용
            hips_pos = df.loc[hips_mask, pos_cols].values
            df.loc[hips_mask, pos_cols] = r_corr.apply(hips_pos)
            
            # Hips 쿼터니언 회전 적용 (글로벌 보정이므로 앞에 곱함)
            hips_quat = df.loc[hips_mask, q_cols].values
            r_hips = R.from_quat(hips_quat)
            df.loc[hips_mask, q_cols] = (r_corr * r_hips).as_quat()

        # 보정 여부와 무관하게 (보정 후) 초기 위치/자세를 확정
        hips_f0 = df[(df['Frame'] == 0) & hips_mask]
        p0 = hips_f0[pos_cols].values[0]

        # ----------------------------------------------------
        # 3. 루트 본(Hips) 위치 영점화
        # ----------------------------------------------------
        df.loc[hips_mask, 'px'] -= p0[0]
        df.loc[hips_mask, 'py'] -= p0[1]
        df.loc[hips_mask, 'pz'] -= p0[2]

        # ----------------------------------------------------
        # 4. 첫 프레임 정면 방향(+Z) 정렬
        # ----------------------------------------------------
        q0_hips = hips_f0[q_cols].values[0]
        r0 = R.from_quat(q0_hips)
        
        yaw0 = r0.as_euler('xyz')[1] 
        r_inv_yaw = R.from_euler('y', -yaw0)
        
        df.loc[hips_mask, pos_cols] = r_inv_yaw.apply(df.loc[hips_mask, pos_cols].values)
        r_hips_final = R.from_quat(df.loc[hips_mask, q_cols].values)
        df.loc[hips_mask, q_cols] = (r_inv_yaw * r_hips_final).as_quat()

        # 무결성 검증을 모두 통과하고 전처리가 완료된 파일만 저장
        df.to_csv(output_path, index=False)
        return True, "성공"

    except Exception as e:
        return False, f"실행 에러: {str(e)}"

def process_dataset_directory(input_base, output_base):
    in_path = Path(input_base)
    out_path = Path(output_base)

    if not in_path.exists():
        print(f"입력 경로를 찾을 수 없습니다: {in_path}")
        return

    csv_files = list(in_path.rglob("*.csv"))
    total_files = len(csv_files)
    
    print(f"총 {total_files}개의 CSV 파일을 검사 및 전처리합니다...\n")

    success_count = 0
    discard_count = 0

    for i, file_path in enumerate(csv_files, 1):
        relative_path = file_path.relative_to(in_path)
        output_file_path = out_path / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        is_success, msg = normalize_motion_file(file_path, output_file_path)
        
        if is_success:
            success_count += 1
        else:
            discard_count += 1
            # 에러가 발생하여 버려진 파일은 저장하지 않고 로그만 남김
            print(f"[{relative_path}] - {msg}")

        if i % 500 == 0 or i == total_files:
            print(f"진행: {i}/{total_files} (성공: {success_count}, 폐기: {discard_count})")

    print("\n" + "="*50)
    print("작업 완료!")
    print(f" - 추출된 정상 파일: {success_count}개")
    print(f" - 폐기된 불량 파일: {discard_count}개")
    print("="*50)

# ==========================================
# 실행 영역
# ==========================================
if __name__ == "__main__":
    INPUT_DIR = r"Sample_Data/Bandai_Dataset_csv_VMC"
    OUTPUT_DIR = r"Sample_Data/Bandai_Dataset_csv_VMC_Normalized"
    process_dataset_directory(INPUT_DIR, OUTPUT_DIR)