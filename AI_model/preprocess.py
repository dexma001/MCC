"""
CSV → .pt 전처리 전용 스크립트 (구 dataset_pipeline.py의 MODE='PREPROCESS').

원본 모션 CSV(Bandai_Dataset_csv_VMC_Normalized)를 읽어
[Frames, 87] = Hips Position(3) + 21관절 Local Quaternion(84) 텐서로 변환해 저장한다.

실행:
    python AI_model/preprocess.py                 # 없는 .pt만 새로 만든다 (기존은 건너뜀)
    python AI_model/preprocess.py --dry-run       # 계획만 출력, 아무것도 쓰지 않음
    python AI_model/preprocess.py --overwrite     # [주의] 기존 .pt를 전부 다시 만든다
    python AI_model/preprocess.py --out-dir other_dir

[주의] 기존 `processed_motions_VMC/`의 .pt는 학습·평가의 기준 데이터다. 기본 실행은 이미 있는
   파일을 건드리지 않으며, 재생성은 `--overwrite`를 명시할 때만 일어난다.
   (2026-08-08 이전 판은 인자를 무시해서 `--help`만 쳐도 전체 전처리가 실행됐다.)

[분리 배경 — 2026-08-07]
  구판은 dataset_pipeline.py 한 파일이 MODE(PREPROCESS/VISUALIZE) × VISUALIZE_TYPE
  (SINGLE/COMPARE)로 3중 분기했고, 모드를 바꾸려면 소스를 편집해야 했다.
  전처리와 시각화는 실행 시점·의존성·목적이 전부 다르므로 파일을 분리한다.
  부수 효과: 이 스크립트는 matplotlib/scipy를 전혀 로드하지 않는다.

[주의] 좌표 규약: 여기서 저장하는 텐서는 원본 CSV와 같은 Unity(Y-Up) 규약이다.
   학습/물리(FK)는 이 규약을 그대로 쓰고, 시각화만 viz_motion.py에서 Z-Up으로 변환한다.
   즉 이 파일은 좌표 변환을 하지 않는다 — 하면 전 파이프라인이 어긋난다.
"""
import argparse
import glob
import os

import pandas as pd
import torch

from dataset_pipeline import BONE_NAMES

# 경로 (프로젝트 루트 실행이 표준, AI_model/ 안에서 실행해도 되도록 폴백)
RAW_CSV_DIR = "Sample_Data/Bandai_Dataset_csv_VMC_Normalized" \
    if os.path.exists("Sample_Data/Bandai_Dataset_csv_VMC_Normalized") \
    else "../Sample_Data/Bandai_Dataset_csv_VMC_Normalized"
PROCESSED_PT_DIR = "processed_motions_VMC" \
    if os.path.exists("processed_motions_VMC") else "../processed_motions_VMC"


def parse_csv_to_quaternion_tensor(motion_csv_path):
    """입력받은 모션 CSV 파일에서 Hips 위치와 전 관절 쿼터니언을 추출합니다."""
    df = pd.read_csv(motion_csv_path)
    frames = sorted(df['Frame'].unique())
    motion_list = []

    for f in frames:
        frame_data = df[df['Frame'] == f].set_index('BoneName')

        # 1. Hips의 동적 Position 추출 [3차원] (cm → m)
        hips_row = frame_data.loc['Hips']
        hips_pos = torch.tensor([hips_row['px'], hips_row['py'], hips_row['pz']],
                                dtype=torch.float32) / 100.0

        # 2. 21개 전 관절의 Quaternion 추출 [21 * 4 = 84차원]
        quats_list = []
        for bone in BONE_NAMES:
            row = frame_data.loc[bone]
            q = torch.tensor([row['qx'], row['qy'], row['qz'], row['qw']], dtype=torch.float32)
            # 안전을 위한 쿼터니언 정규화
            q = q / (torch.norm(q) + 1e-8)
            quats_list.append(q)

        quats_tensor = torch.cat(quats_list)   # [84]

        # 3. [3 + 84 = 87] 차원으로 결합
        motion_list.append(torch.cat([hips_pos, quats_tensor]))

    return torch.stack(motion_list)


def run_preprocess(raw_csv_dir=RAW_CSV_DIR, out_dir=PROCESSED_PT_DIR,
                   overwrite=False, dry_run=False):
    """
    raw_csv_dir의 모든 CSV를 [Frames, 87] 텐서로 변환해 out_dir에 저장.

    [주의] 기본값은 '보수적'이다 — out_dir에 이미 있는 .pt는 건너뛴다.
       기존 산출물은 학습·평가의 기준 데이터이므로, 덮어쓰려면 overwrite=True를
       명시해야 한다 (CLI: --overwrite). dry_run=True면 아무것도 쓰지 않고 계획만 출력.

    반환: (변환한 수, 건너뛴 수)
    """
    if not os.path.isdir(raw_csv_dir):
        print(f"❌ 원본 CSV 폴더를 찾을 수 없습니다: {raw_csv_dir}")
        return 0, 0

    csv_files = sorted(glob.glob(os.path.join(raw_csv_dir, "**", "*.csv"), recursive=True))
    if not csv_files:
        print(f"❌ 변환할 CSV가 없습니다: {raw_csv_dir}")
        return 0, 0

    def out_path(csv_path):
        return os.path.join(out_dir, os.path.basename(csv_path).replace(".csv", ".pt"))

    existing = [p for p in csv_files if os.path.exists(out_path(p))]
    todo = csv_files if overwrite else [p for p in csv_files if not os.path.exists(out_path(p))]

    print(f"원본 CSV {len(csv_files)}개 | 이미 존재하는 .pt {len(existing)}개 → {out_dir}")
    if existing and not overwrite:
        print(f"⏭️  기존 {len(existing)}개는 건너뜁니다 (덮어쓰려면 --overwrite).")
    if existing and overwrite:
        print(f"⚠️  --overwrite: 기존 {len(existing)}개를 다시 씁니다.")
    if not todo:
        print("✅ 새로 변환할 파일이 없습니다. (아무것도 쓰지 않음)")
        return 0, len(existing)

    if dry_run:
        print(f"🧪 --dry-run: {len(todo)}개를 변환할 예정이지만 아무것도 쓰지 않았습니다.")
        for p in todo[:5]:
            print(f"     {os.path.basename(p)} → {os.path.basename(out_path(p))}")
        if len(todo) > 5:
            print(f"     ... 외 {len(todo)-5}개")
        return 0, len(csv_files) - len(todo)

    os.makedirs(out_dir, exist_ok=True)
    print(f"🔄 전처리 시작: {len(todo)}개를 변환합니다.")
    for csv_path in todo:
        motion_tensor = parse_csv_to_quaternion_tensor(csv_path)
        torch.save(motion_tensor, out_path(csv_path))
    n_skip = len(csv_files) - len(todo)
    print(f"✅ 변환 완료: {len(todo)}개 (건너뜀 {n_skip}개) → {out_dir}")
    return len(todo), n_skip


def main(argv=None):
    """
    CLI 진입점.

    [주의] 이 함수가 존재하는 이유(2026-08-08): 구판은 `__main__`에서 곧바로
    run_preprocess()를 호출해 **어떤 인자를 줘도 전처리가 실행됐다.**
    실제로 `--help`를 친 것만으로 3077개 파일 전처리가 시작된 사고가 있었다.
    이제 인자를 해석하고, 기존 산출물은 --overwrite 없이는 건드리지 않는다.
    """
    p = argparse.ArgumentParser(
        description="원본 모션 CSV → [Frames, 87] .pt 텐서 변환. "
                    "기본값은 기존 .pt를 건너뛴다(덮어쓰기 방지).")
    p.add_argument("--raw-csv-dir", default=RAW_CSV_DIR, help="원본 CSV 폴더")
    p.add_argument("--out-dir", default=PROCESSED_PT_DIR, help=".pt 저장 폴더")
    p.add_argument("--overwrite", action="store_true",
                   help="기존 .pt를 다시 쓴다 (기본: 건너뜀). 학습 기준 데이터를 "
                        "재생성하는 것이므로 의도적으로만 사용할 것.")
    p.add_argument("--dry-run", action="store_true",
                   help="무엇을 변환할지만 출력하고 아무것도 쓰지 않는다.")
    args = p.parse_args(argv)

    run_preprocess(raw_csv_dir=args.raw_csv_dir, out_dir=args.out_dir,
                   overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
