import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
FIG_DIR       = Path(__file__).parent.parent / "reports" / "figures"

plt.rcParams["font.family"]        = ["NanumSquare", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"]         = 130

C0 = "#54A082"
C1 = "#D4785A"
C2 = "#5A8AD4"


def find_best_threshold(y_true, y_proba, target_recall=None):
    """
    threshold 후보 범위에서 각 지표 계산.
    target_recall 지정 시 해당 Recall 이상인 구간에서 F1 최대 threshold 반환.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "recall":    recall_score(y_true, pred, zero_division=0),
            "precision": precision_score(y_true, pred, zero_division=0),
            "f1":        f1_score(y_true, pred, zero_division=0),
        })
    df = pd.DataFrame(rows)

    if target_recall is not None:
        cand = df[df["recall"] >= target_recall]
        if len(cand) == 0:
            best = df.loc[df["recall"].idxmax()]
        else:
            best = cand.loc[cand["f1"].idxmax()]
    else:
        best = df.loc[df["f1"].idxmax()]

    return df, best


def plot_threshold(df, best_row, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13)

    # 왼쪽: Recall / Precision / F1 vs threshold
    ax = axes[0]
    ax.plot(df["threshold"], df["recall"],    color=C1, lw=2, label="Recall")
    ax.plot(df["threshold"], df["precision"], color=C0, lw=2, label="Precision")
    ax.plot(df["threshold"], df["f1"],        color=C2, lw=2, label="F1")
    ax.axvline(best_row["threshold"], color="black", linestyle="--", lw=1.2,
               label=f'최적 threshold={best_row["threshold"]:.2f}')
    ax.axhline(best_row["recall"],    color=C1, linestyle=":", lw=1, alpha=0.6)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_title("Threshold vs 지표")

    # 오른쪽: Recall vs Precision (PR curve처럼)
    ax2 = axes[1]
    ax2.plot(df["recall"], df["precision"], color=C0, lw=2)
    ax2.scatter([best_row["recall"]], [best_row["precision"]],
                color="black", s=80, zorder=5,
                label=f'최적점  Recall={best_row["recall"]:.3f}  Prec={best_row["precision"]:.3f}')
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.set_title("Recall vs Precision")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"저장: {save_path.name}")


# %%
# Task A XGBoost
oof_xgb_a = pd.read_csv(PROCESSED_DIR / "oof_xgb_task_a.csv")
y_a        = oof_xgb_a["task_a"].values
proba_xgb_a = oof_xgb_a["xgb_proba"].values

print("=" * 60)
print(" Task A (가공불량) - XGBoost Threshold 최적화")
print("=" * 60)

# 기본값 0.5 성능
pred_05 = (proba_xgb_a >= 0.5).astype(int)
print(f"\n[현재 threshold=0.50]")
print(f"  Recall={recall_score(y_a, pred_05):.3f}  "
      f"Precision={precision_score(y_a, pred_05):.3f}  "
      f"F1={f1_score(y_a, pred_05):.3f}")

# Recall >= 0.85 조건에서 F1 최대
df_a, best_a_r85 = find_best_threshold(y_a, proba_xgb_a, target_recall=0.85)
print(f"\n[Recall ≥ 0.85 조건 최적 threshold]")
print(f"  threshold={best_a_r85['threshold']:.2f}  "
      f"Recall={best_a_r85['recall']:.3f}  "
      f"Precision={best_a_r85['precision']:.3f}  "
      f"F1={best_a_r85['f1']:.3f}")

# F1 최대 threshold
_, best_a_f1 = find_best_threshold(y_a, proba_xgb_a)
print(f"\n[F1 최대 threshold]")
print(f"  threshold={best_a_f1['threshold']:.2f}  "
      f"Recall={best_a_f1['recall']:.3f}  "
      f"Precision={best_a_f1['precision']:.3f}  "
      f"F1={best_a_f1['f1']:.3f}")

print(f"\n--- threshold별 전체 표 (주요 구간) ---")
print(df_a[df_a["threshold"].between(0.1, 0.7)].to_string(index=False))

plot_threshold(df_a, best_a_r85,
               title="Task A XGBoost: Threshold 최적화 (Recall ≥ 0.85 기준)",
               save_path=FIG_DIR / "24_threshold_task_a_xgb.png")


# %%
# Task A DNN
oof_dnn_a  = pd.read_csv(PROCESSED_DIR / "oof_dnn_task_a.csv")
proba_dnn_a = oof_dnn_a["dnn_proba"].values

print("\n" + "=" * 60)
print(" Task A (가공불량) - DNN Threshold 최적화")
print("=" * 60)

pred_05_dnn = (proba_dnn_a >= 0.5).astype(int)
print(f"\n[현재 threshold=0.50]")
print(f"  Recall={recall_score(y_a, pred_05_dnn):.3f}  "
      f"Precision={precision_score(y_a, pred_05_dnn):.3f}  "
      f"F1={f1_score(y_a, pred_05_dnn):.3f}")

df_dnn_a, best_dnn_a_r85 = find_best_threshold(y_a, proba_dnn_a, target_recall=0.85)
print(f"\n[Recall ≥ 0.85 조건 최적 threshold]")
print(f"  threshold={best_dnn_a_r85['threshold']:.2f}  "
      f"Recall={best_dnn_a_r85['recall']:.3f}  "
      f"Precision={best_dnn_a_r85['precision']:.3f}  "
      f"F1={best_dnn_a_r85['f1']:.3f}")

plot_threshold(df_dnn_a, best_dnn_a_r85,
               title="Task A DNN: Threshold 최적화 (Recall ≥ 0.85 기준)",
               save_path=FIG_DIR / "25_threshold_task_a_dnn.png")


# %%
# Task B XGBoost
oof_xgb_b  = pd.read_csv(PROCESSED_DIR / "oof_xgb_task_b.csv")
y_b         = oof_xgb_b["task_b"].values
proba_xgb_b = oof_xgb_b["xgb_proba"].values

print("\n" + "=" * 60)
print(" Task B (공구마모) - XGBoost Threshold 최적화")
print("=" * 60)

pred_05_b = (proba_xgb_b >= 0.5).astype(int)
print(f"\n[현재 threshold=0.50]")
print(f"  Recall={recall_score(y_b, pred_05_b):.3f}  "
      f"Precision={precision_score(y_b, pred_05_b):.3f}  "
      f"F1={f1_score(y_b, pred_05_b):.3f}")

df_b, best_b_r75 = find_best_threshold(y_b, proba_xgb_b, target_recall=0.75)
print(f"\n[Recall ≥ 0.75 조건 최적 threshold]")
print(f"  threshold={best_b_r75['threshold']:.2f}  "
      f"Recall={best_b_r75['recall']:.3f}  "
      f"Precision={best_b_r75['precision']:.3f}  "
      f"F1={best_b_r75['f1']:.3f}")

_, best_b_f1 = find_best_threshold(y_b, proba_xgb_b)
print(f"\n[F1 최대 threshold]")
print(f"  threshold={best_b_f1['threshold']:.2f}  "
      f"Recall={best_b_f1['recall']:.3f}  "
      f"Precision={best_b_f1['precision']:.3f}  "
      f"F1={best_b_f1['f1']:.3f}")

print(f"\n--- threshold별 전체 표 (주요 구간) ---")
print(df_b[df_b["threshold"].between(0.1, 0.7)].to_string(index=False))

plot_threshold(df_b, best_b_r75,
               title="Task B XGBoost: Threshold 최적화 (Recall ≥ 0.75 기준)",
               save_path=FIG_DIR / "26_threshold_task_b_xgb.png")

print("\n=== Threshold 최적화 완료 ===")
