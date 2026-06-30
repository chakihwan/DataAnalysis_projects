import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
FIG_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib.font_manager as fm

plt.rcParams["font.family"]        = "NanumSquare"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"]         = 130
plt.rcParams["figure.figsize"]     = (11, 6)
plt.rcParams["axes.facecolor"]     = "white"
plt.rcParams["figure.facecolor"]   = "white"
plt.rcParams["font.size"]          = 11
plt.rcParams["axes.titlesize"]     = 13
plt.rcParams["axes.titleweight"]   = "normal"
plt.rcParams["axes.labelsize"]     = 11

C0 = "#54A082"   # 뮤트 틸 그린
C1 = "#D4785A"   # 테라코타 오렌지
PALETTE = {"양품(0)": C0, "불량(1)": C1,
           "unworn(0)": C0, "worn(1)": C1}
ALPHA = 0.85

sns.set_theme(style="whitegrid", font="NanumSquare")
sns.set_context("notebook", font_scale=1.05)
plt.rcParams["grid.color"]     = "#E5E5E5"
plt.rcParams["grid.linewidth"] = 0.8

# %%
# 데이터 로드 및 타겟 생성
train = pd.read_csv(RAW_DIR / "train.csv")

good = (train["machining_finalized"] == "yes") & (train["passed_visual_inspection"] == "yes")
train["task_a"] = (~good).astype(int)
train["task_b"] = (train["tool_condition"] == "worn").astype(int)

print("=== train.csv ===")
print(train[["No", "feedrate", "clamp_pressure", "tool_condition", "task_a", "task_b"]])

# %%
# [1] 타겟 분포
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

labels_a = ["양품(0)", "불량(1)"]
vals_a = train["task_a"].value_counts().sort_index().values
axes[0].bar(labels_a, vals_a, color=[PALETTE["양품(0)"], PALETTE["불량(1)"]], alpha=ALPHA, width=0.5)
axes[0].set_title("Task A: 가공불량 분포")
axes[0].set_ylabel("실험 수")
for i, v in enumerate(vals_a):
    axes[0].text(i, v + 0.1, str(v), ha="center", fontweight="bold")

labels_b = ["unworn(0)", "worn(1)"]
vals_b = train["task_b"].value_counts().sort_index().values
axes[1].bar(labels_b, vals_b, color=[PALETTE["unworn(0)"], PALETTE["worn(1)"]], alpha=ALPHA, width=0.5)
axes[1].set_title("Task B: 공구마모 분포")
axes[1].set_ylabel("실험 수")
for i, v in enumerate(vals_b):
    axes[1].text(i, v + 0.1, str(v), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig(FIG_DIR / "01_target_distribution.png")
plt.show()
print("저장: 01_target_distribution.png")

# %%
# [2] train.csv 피처 분포 (feedrate, clamp_pressure)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

train["feedrate"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color=C0, alpha=ALPHA)
axes[0].set_title("feedrate 분포")
axes[0].set_xlabel("feedrate (mm/min)")
axes[0].set_ylabel("실험 수")
axes[0].tick_params(axis="x", rotation=0)

train["clamp_pressure"].value_counts().sort_index().plot(kind="bar", ax=axes[1], color=C0, alpha=ALPHA)
axes[1].set_title("clamp_pressure 분포")
axes[1].set_xlabel("clamp_pressure (bar)")
axes[1].set_ylabel("실험 수")
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(FIG_DIR / "02_feature_distribution.png")
plt.show()
print("저장: 02_feature_distribution.png")

# %%
# [3] 피처 vs 타겟 관계 (Task A)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

train["task_a_label"] = train["task_a"].map({0: "양품(0)", 1: "불량(1)"})
sns.countplot(data=train, x="feedrate", hue="task_a_label",
              palette=PALETTE, saturation=1, ax=axes[0])
axes[0].set_title("feedrate vs Task A (불량여부)")
axes[0].set_xlabel("feedrate (mm/min)")

sns.countplot(data=train, x="clamp_pressure", hue="task_a_label",
              palette=PALETTE, saturation=1, ax=axes[1])
axes[1].set_title("clamp_pressure vs Task A (불량여부)")
axes[1].set_xlabel("clamp_pressure (bar)")

sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "03_feature_vs_task_a.png")
plt.show()
print("저장: 03_feature_vs_task_a.png")

# %%
# [4] 피처 vs 타겟 관계 (Task B)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

train["task_b_label"] = train["task_b"].map({0: "unworn(0)", 1: "worn(1)"})
sns.countplot(data=train, x="feedrate", hue="task_b_label",
              palette=PALETTE, saturation=1, ax=axes[0])
axes[0].set_title("feedrate vs Task B (공구마모)")
axes[0].set_xlabel("feedrate (mm/min)")

sns.countplot(data=train, x="clamp_pressure", hue="task_b_label",
              palette=PALETTE, saturation=1, ax=axes[1])
axes[1].set_title("clamp_pressure vs Task B (공구마모)")
axes[1].set_xlabel("clamp_pressure (bar)")

sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "04_feature_vs_task_b.png")
plt.show()
print("저장: 04_feature_vs_task_b.png")

# %%
# [5] 실험 센서 데이터 집계 - 전체 통계량 + 공정 단계별 통계량
SENSOR_COLS = [
    "X_OutputCurrent", "X_OutputVoltage", "X_OutputPower",
    "Y_OutputCurrent", "Y_OutputVoltage", "Y_OutputPower",
    "Z_OutputCurrent", "Z_OutputVoltage",
    "S_OutputCurrent", "S_OutputVoltage", "S_OutputPower", "S_SystemInertia",
    "X_ActualVelocity", "Y_ActualVelocity", "Z_ActualVelocity", "S_ActualVelocity",
]

CUTTING_STAGES = ["Layer 1 Up", "Layer 1 Down",
                  "Layer 2 Up", "Layer 2 Down",
                  "Layer 3 Up", "Layer 3 Down"]

records = []
for i in range(1, 26):
    df = pd.read_csv(RAW_DIR / f"experiment_{i:02d}.csv")
    row = {"No": i}

    # 전체 통계량
    for col in SENSOR_COLS:
        row[f"{col}_mean"] = df[col].mean()
        row[f"{col}_std"]  = df[col].std()
        row[f"{col}_max"]  = df[col].max()

    # 절삭 구간(Layer)만 필터 통계량
    cutting = df[df["Machining_Process"].isin(CUTTING_STAGES)]
    for col in SENSOR_COLS:
        row[f"{col}_cut_mean"] = cutting[col].mean()
        row[f"{col}_cut_std"]  = cutting[col].std()

    records.append(row)

sensor_stats = pd.DataFrame(records)
merged = pd.merge(train, sensor_stats, on="No")

print(f"집계 완료: {sensor_stats.shape[1]}개 피처")
print(merged[["No", "task_a", "task_b",
              "X_OutputCurrent_mean", "S_OutputCurrent_mean"]].head())

# %%
# [6] 핵심 센서 - Task A 박스플롯 (양품 vs 불량)
KEY_SENSORS = [
    "X_OutputCurrent_mean", "Y_OutputCurrent_mean",
    "Z_OutputCurrent_mean", "S_OutputCurrent_mean",
    "S_OutputPower_mean",   "S_SystemInertia_mean",
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

merged["task_a_label"] = merged["task_a"].map({0: "양품(0)", 1: "불량(1)"})
for idx, col in enumerate(KEY_SENSORS):
    sns.boxplot(data=merged, x="task_a_label", y=col,
                hue="task_a_label",
                palette=PALETTE,
                legend=False, ax=axes[idx])
    axes[idx].set_title(col.replace("_mean", ""))
    axes[idx].set_xlabel("")

plt.suptitle("Task A: 양품 vs 불량 - 핵심 센서 분포", fontsize=13, y=1.01)
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "05_sensor_boxplot_task_a.png", bbox_inches="tight")
plt.show()
print("저장: 05_sensor_boxplot_task_a.png")

# %%
# [7] 핵심 센서 - Task B 박스플롯 (unworn vs worn)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

merged["task_b_label"] = merged["task_b"].map({0: "unworn(0)", 1: "worn(1)"})
for idx, col in enumerate(KEY_SENSORS):
    sns.boxplot(data=merged, x="task_b_label", y=col,
                hue="task_b_label",
                palette=PALETTE,
                legend=False, ax=axes[idx])
    axes[idx].set_title(col.replace("_mean", ""))
    axes[idx].set_xlabel("")

plt.suptitle("Task B: unworn vs worn - 핵심 센서 분포", fontsize=13, y=1.01)
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "06_sensor_boxplot_task_b.png", bbox_inches="tight")
plt.show()
print("저장: 06_sensor_boxplot_task_b.png")

# %%
# [8] Machining_Process 단계별 센서 평균 (실험 전체 합산)
all_exp = []
for i in range(1, 26):
    df = pd.read_csv(RAW_DIR / f"experiment_{i:02d}.csv")
    df["exp_no"] = i
    all_exp.append(df)
all_data = pd.concat(all_exp, ignore_index=True)

stage_order = ["Starting", "Prep", "Repositioning",
               "Layer 1 Up", "Layer 1 Down",
               "Layer 2 Up", "Layer 2 Down",
               "Layer 3 Up", "Layer 3 Down", "end"]

stage_avg = (all_data.groupby("Machining_Process")[
    ["S_OutputCurrent", "S_OutputPower", "X_OutputCurrent", "Y_OutputCurrent"]]
    .mean()
    .reindex([s for s in stage_order if s in all_data["Machining_Process"].unique()])
)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for idx, col in enumerate(stage_avg.columns):
    stage_avg[col].plot(kind="bar", ax=axes[idx], color=C0, alpha=ALPHA)
    axes[idx].set_title(f"공정 단계별 {col} 평균")
    axes[idx].tick_params(axis="x", rotation=45)
    axes[idx].set_xlabel("")

plt.suptitle("Machining_Process 단계별 센서 평균", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "07_stage_sensor_avg.png", bbox_inches="tight")
plt.show()
print("저장: 07_stage_sensor_avg.png")

# %%
# [9] 상관관계 히트맵 (집계 피처 × 타겟)
corr_cols = [c for c in merged.columns
             if c.endswith("_mean") and "cut" not in c] + ["task_a", "task_b"]
corr_matrix = merged[corr_cols].corr()[["task_a", "task_b"]].drop(["task_a", "task_b"])
corr_matrix = corr_matrix.sort_values("task_a", ascending=False)

fig, ax = plt.subplots(figsize=(6, 14))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("센서 통계량 vs 타겟 상관계수")

plt.tight_layout()
plt.savefig(FIG_DIR / "08_correlation_heatmap.png", bbox_inches="tight")
plt.show()
print("저장: 08_correlation_heatmap.png")

# %%
# [10] 절삭 구간 전류 분포 - 양품 vs 불량 (Task A)
cut_sensor = [c for c in merged.columns if c.endswith("_cut_mean")]
cut_current = [c for c in cut_sensor if "Current" in c]

fig, axes = plt.subplots(1, len(cut_current), figsize=(14, 5))
for idx, col in enumerate(cut_current):
    sns.boxplot(data=merged, x="task_a_label", y=col,
                hue="task_a_label",
                palette=PALETTE,
                legend=False, ax=axes[idx])
    axes[idx].set_title(col.replace("_cut_mean", "\n절삭구간"))
    axes[idx].set_xlabel("")

plt.suptitle("Task A: 절삭 구간(Layer) 전류 비교", fontsize=13, y=1.01)
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "09_cutting_current_task_a.png", bbox_inches="tight")
plt.show()
print("저장: 09_cutting_current_task_a.png")

print("\n=== EDA 완료 ===")
print(f"생성된 figure: {len(list(FIG_DIR.glob('*.png')))}개")
print(f"저장 위치: {FIG_DIR}")
