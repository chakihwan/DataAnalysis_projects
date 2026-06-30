import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (recall_score, precision_score, f1_score,
                              roc_auc_score, confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
FIG_DIR       = Path(__file__).parent.parent / "reports" / "figures"

plt.rcParams["font.family"]        = ["NanumSquare", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"]         = 130

C0 = "#54A082"   # teal (XGBoost)
C1 = "#D4785A"   # terracotta (DNN)


# %%
# OOF 결과 로드
oof_xgb_a = pd.read_csv(PROCESSED_DIR / "oof_xgb_task_a.csv")
oof_xgb_b = pd.read_csv(PROCESSED_DIR / "oof_xgb_task_b.csv")
oof_dnn_a = pd.read_csv(PROCESSED_DIR / "oof_dnn_task_a.csv")
oof_dnn_b = pd.read_csv(PROCESSED_DIR / "oof_dnn_task_b.csv")

y_a = oof_xgb_a["task_a"].values
y_b = oof_xgb_b["task_b"].values


def calc_metrics(y_true, y_pred, y_proba):
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
    return {
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 3),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 3),
        "AUC":       round(auc, 3),
    }


# %%
# 종합 성능 비교표 출력
rows = []
for task, y_true, oof_xgb, oof_dnn, pos_label in [
    ("Task A (가공불량)", y_a, oof_xgb_a, oof_dnn_a, "불량(1)"),
    ("Task B (공구마모)", y_b, oof_xgb_b, oof_dnn_b, "worn(1)"),
]:
    m_xgb = calc_metrics(y_true, oof_xgb["xgb_pred"].values, oof_xgb["xgb_proba"].values)
    m_dnn = calc_metrics(y_true, oof_dnn["dnn_pred"].values, oof_dnn["dnn_proba"].values)
    rows.append({"Task": task, "Model": "XGBoost", **m_xgb})
    rows.append({"Task": task, "Model": "DNN",     **m_dnn})

df_cmp = pd.DataFrame(rows)
print("\n" + "=" * 60)
print(" XGBoost vs DNN 성능 비교 (OOF 5-Fold)")
print("=" * 60)
print(df_cmp.to_string(index=False))


# %%
# 그림 20: 성능 비교 바 차트 (Task A / Task B × 4개 지표)
metrics_list = ["Recall", "Precision", "F1", "AUC"]
task_names   = ["Task A (가공불량)", "Task B (공구마모)"]

fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
fig.suptitle("XGBoost vs DNN 성능 비교 (OOF 5-Fold)", fontsize=14, y=1.01)

for t_idx, task in enumerate(task_names):
    sub = df_cmp[df_cmp["Task"] == task]
    for m_idx, metric in enumerate(metrics_list):
        ax = axes[t_idx][m_idx]
        vals   = sub[metric].values
        models = sub["Model"].values
        colors = [C0, C1]
        bars   = ax.bar(models, vals, color=colors, width=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{metric}", fontsize=11)
        if m_idx == 0:
            ax.set_ylabel(task.split(" ")[0] + " " + task.split(" ")[1], fontsize=10)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.03, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "20_model_comparison_bar.png", bbox_inches="tight")
plt.close()
print("\n저장: 20_model_comparison_bar.png")


# %%
# 그림 21: ROC Curve 비교 (Task A / Task B 나란히)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, task_label, y_true, oof_xgb, oof_dnn in [
    (axes[0], "Task A (가공불량)", y_a, oof_xgb_a, oof_dnn_a),
    (axes[1], "Task B (공구마모)", y_b, oof_xgb_b, oof_dnn_b),
]:
    for proba_col, model_name, color in [
        ("xgb_proba", "XGBoost", C0),
        ("dnn_proba", "DNN",     C1),
    ]:
        src = oof_xgb if "xgb" in proba_col else oof_dnn
        fpr, tpr, _ = roc_curve(y_true, src[proba_col].values)
        auc = roc_auc_score(y_true, src[proba_col].values)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve: {task_label}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

plt.tight_layout()
plt.savefig(FIG_DIR / "21_roc_curve_comparison.png", bbox_inches="tight")
plt.close()
print("저장: 21_roc_curve_comparison.png")


# %%
# 그림 22: Confusion Matrix 4개 나란히 (XGBoost/DNN × Task A/B)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Confusion Matrix 비교 (OOF 5-Fold)", fontsize=13)

configs = [
    (axes[0][0], "Task A — XGBoost", y_a, oof_xgb_a["xgb_pred"].values, ["양품(0)", "불량(1)"]),
    (axes[0][1], "Task A — DNN",     y_a, oof_dnn_a["dnn_pred"].values,  ["양품(0)", "불량(1)"]),
    (axes[1][0], "Task B — XGBoost", y_b, oof_xgb_b["xgb_pred"].values, ["unworn(0)", "worn(1)"]),
    (axes[1][1], "Task B — DNN",     y_b, oof_dnn_b["dnn_pred"].values,  ["unworn(0)", "worn(1)"]),
]

for ax, title, y_true, y_pred, labels in configs:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    ax.set_title(title, fontsize=11)

plt.tight_layout()
plt.savefig(FIG_DIR / "22_confusion_matrix_all.png", bbox_inches="tight")
plt.close()
print("저장: 22_confusion_matrix_all.png")


# %%
# 샘플별 예측 확률 비교 (XGBoost vs DNN proba scatter)
# 그림 23: Task A / Task B proba scatter
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, task_label, y_true, oof_xgb, oof_dnn in [
    (axes[0], "Task A (가공불량)", y_a, oof_xgb_a, oof_dnn_a),
    (axes[1], "Task B (공구마모)", y_b, oof_xgb_b, oof_dnn_b),
]:
    xgb_p = oof_xgb["xgb_proba"].values
    dnn_p = oof_dnn["dnn_proba"].values

    for cls, label, marker in [(0, "음성(양품/unworn)", "o"), (1, "양성(불량/worn)", "^")]:
        mask = y_true == cls
        ax.scatter(xgb_p[mask], dnn_p[mask],
                   color=(C0 if cls == 0 else C1),
                   marker=marker, s=70, alpha=0.8, label=label, zorder=3)

    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("XGBoost 예측 확률")
    ax.set_ylabel("DNN 예측 확률")
    ax.set_title(f"{task_label}: XGBoost vs DNN 예측 확률")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(FIG_DIR / "23_proba_scatter.png", bbox_inches="tight")
plt.close()
print("저장: 23_proba_scatter.png")


# %%
# 최종 요약 출력
print("\n" + "=" * 60)
print(" 최종 모델 선정 권고")
print("=" * 60)
for task, y_true, oof_xgb, oof_dnn in [
    ("Task A (가공불량)", y_a, oof_xgb_a, oof_dnn_a),
    ("Task B (공구마모)", y_b, oof_xgb_b, oof_dnn_b),
]:
    m_x = calc_metrics(y_true, oof_xgb["xgb_pred"].values, oof_xgb["xgb_proba"].values)
    m_d = calc_metrics(y_true, oof_dnn["dnn_pred"].values, oof_dnn["dnn_proba"].values)
    winner = "XGBoost" if m_x["Recall"] >= m_d["Recall"] else "DNN"
    print(f"\n[{task}]")
    print(f"  XGBoost: Recall={m_x['Recall']:.3f} / F1={m_x['F1']:.3f} / AUC={m_x['AUC']:.3f}")
    print(f"  DNN:     Recall={m_d['Recall']:.3f} / F1={m_d['F1']:.3f} / AUC={m_d['AUC']:.3f}")
    print(f"  → 권고 모델: {winner} (Recall 기준)")

print("\n=== 평가 완료 ===")
