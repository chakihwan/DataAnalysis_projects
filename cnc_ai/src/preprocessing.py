import pandas as pd
import numpy as np
import os
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_COLS = [
    "X_OutputCurrent", "X_OutputVoltage", "X_OutputPower",
    "Y_OutputCurrent", "Y_OutputVoltage", "Y_OutputPower",
    "Z_OutputCurrent", "Z_OutputVoltage",
    "S_OutputCurrent", "S_OutputVoltage", "S_OutputPower", "S_SystemInertia",
    "X_ActualVelocity", "Y_ActualVelocity", "Z_ActualVelocity", "S_ActualVelocity",
]

LAYER1 = ["Layer 1 Up", "Layer 1 Down"]
LAYER2 = ["Layer 2 Up", "Layer 2 Down"]
LAYER3 = ["Layer 3 Up", "Layer 3 Down"]
CUTTING_STAGES = LAYER1 + LAYER2 + LAYER3

# %%
# 타겟 생성
train = pd.read_csv(RAW_DIR / "train.csv")

good = (train["machining_finalized"] == "yes") & (train["passed_visual_inspection"] == "yes")
train["task_a"] = (~good).astype(int)
train["task_b"] = (train["tool_condition"] == "worn").astype(int)

print("=== 타겟 분포 ===")
print(f"Task A  양품: {(train['task_a']==0).sum()}  불량: {(train['task_a']==1).sum()}")
print(f"Task B  unworn: {(train['task_b']==0).sum()}  worn: {(train['task_b']==1).sum()}")

# %%
# 실험별 센서 집계
# 집계 단위:
#   1) 전체 구간: mean / std / max
#   2) 절삭 구간(Layer 전체): cut_mean / cut_std
#   3) 레이어별(Layer1/2/3): layer{n}_mean / layer{n}_std  ← stage 변화 피처
# row_count: Task A 전용 피처 (불량 실험은 가공 중단으로 행이 적음)

records = []
for i in range(1, 26):
    df = pd.read_csv(RAW_DIR / f"experiment_{i:02d}.csv")
    row = {"No": i, "row_count": len(df)}

    valid_cols = [c for c in SENSOR_COLS if c in df.columns]

    # 전체 구간
    for col in valid_cols:
        row[f"{col}_mean"] = df[col].mean()
        row[f"{col}_std"]  = df[col].std()
        row[f"{col}_max"]  = df[col].max()

    # 절삭 구간 전체 (Layer 1~3)
    cut = df[df["Machining_Process"].isin(CUTTING_STAGES)]
    for col in valid_cols:
        row[f"{col}_cut_mean"] = cut[col].mean() if len(cut) > 0 else np.nan
        row[f"{col}_cut_std"]  = cut[col].std()  if len(cut) > 0 else np.nan

    # 레이어별
    for layer_name, stages in [("layer1", LAYER1), ("layer2", LAYER2), ("layer3", LAYER3)]:
        seg = df[df["Machining_Process"].isin(stages)]
        for col in valid_cols:
            row[f"{col}_{layer_name}_mean"] = seg[col].mean() if len(seg) > 0 else np.nan
            row[f"{col}_{layer_name}_std"]  = seg[col].std()  if len(seg) > 0 else np.nan

    records.append(row)

sensor_features = pd.DataFrame(records)
print(f"\n집계 완료: {len(records)}개 실험 / {sensor_features.shape[1]-1}개 피처 (No 제외)")

# %%
# 타겟 병합 및 태스크별 데이터셋 생성
meta_cols = ["No", "feedrate", "clamp_pressure"]
merged = pd.merge(
    train[meta_cols + ["task_a", "task_b"]],
    sensor_features,
    on="No"
)

sensor_feat_cols = [c for c in sensor_features.columns if c != "No"]

# Task A: row_count 포함 (불량 실험 조기 종료 패턴)
feat_cols_a = sensor_feat_cols  # row_count 포함
dataset_a = merged[meta_cols + feat_cols_a + ["task_a"]].copy()

# Task B: row_count 제외 (가공 완료 여부와 무관)
feat_cols_b = [c for c in sensor_feat_cols if c != "row_count"]
dataset_b = merged[meta_cols + feat_cols_b + ["task_b"]].copy()

print(f"\nTask A 데이터셋: {dataset_a.shape[0]}행 × {dataset_a.shape[1]}열")
print(f"  피처 {dataset_a.shape[1] - len(meta_cols) - 1}개 + meta {len(meta_cols)}개 + target 1개")
print(f"Task B 데이터셋: {dataset_b.shape[0]}행 × {dataset_b.shape[1]}열")
print(f"  피처 {dataset_b.shape[1] - len(meta_cols) - 1}개 + meta {len(meta_cols)}개 + target 1개")

# %%
# 저장 (기존 파일이 다른 프로세스에 잠겨있을 경우 tmp 경유 교체)
def safe_save(df, path):
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    if path.exists():
        try:
            path.unlink()
        except PermissionError:
            pass  # 잠겨있으면 os.replace로 시도
    os.replace(tmp, path)
    print(f"  저장: {path.name}")

safe_save(dataset_a, PROCESSED_DIR / "dataset_task_a.csv")
safe_save(dataset_b, PROCESSED_DIR / "dataset_task_b.csv")

# %%
# 검증
print("\n=== 결측치 확인 ===")
missing_a = dataset_a.isnull().sum()
missing_b = dataset_b.isnull().sum()
print(f"Task A 결측: {missing_a[missing_a > 0].to_dict() or '없음'}")
print(f"Task B 결측: {missing_b[missing_b > 0].to_dict() or '없음'}")

print("\n=== Task A 핵심 피처 평균 (양품 vs 불량) ===")
key_a = ["row_count", "S_OutputPower_mean", "S_ActualVelocity_mean", "S_OutputVoltage_mean"]
print(dataset_a.groupby("task_a")[key_a].mean().round(3))

print("\n=== Task B 핵심 피처 평균 (unworn vs worn) ===")
key_b = ["Y_OutputVoltage_mean", "X_OutputVoltage_mean", "S_SystemInertia_mean"]
print(dataset_b.groupby("task_b")[key_b].mean().round(3))

