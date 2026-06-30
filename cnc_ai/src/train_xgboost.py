import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (recall_score, precision_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report)
import json
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models"
FIG_DIR       = Path(__file__).parent.parent / "reports" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"]        = "NanumSquare"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"]         = 130

C0 = "#54A082"
C1 = "#D4785A"
RANDOM_STATE = 42
META_COLS    = ["No", "feedrate", "clamp_pressure"]

# 샘플 수(25)가 적어 shallow tree + 강한 정규화
XGB_PARAMS = dict(
    n_estimators      = 100,
    max_depth         = 3,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.5,
    min_child_weight  = 2,
    reg_alpha         = 1.0,
    reg_lambda        = 2.0,
    eval_metric       = "logloss",
    random_state      = RANDOM_STATE,
    verbosity         = 0,
)

# %%
# 공통 CV 함수
def run_cv(X, y, scale_pos_weight, task_name):
    # X는 DataFrame으로 받아야 피처명이 모델에 기록됨
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = []
    oof_preds = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros(len(y))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_val)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        oof_preds[val_idx] = pred
        oof_proba[val_idx] = proba

        auc = roc_auc_score(y_val, proba) if len(np.unique(y_val)) > 1 else np.nan
        fold_metrics.append({
            "fold":      fold,
            "recall":    recall_score(y_val, pred, zero_division=0),
            "precision": precision_score(y_val, pred, zero_division=0),
            "f1":        f1_score(y_val, pred, zero_division=0),
            "auc":       auc,
        })

    df_m = pd.DataFrame(fold_metrics)

    print(f"\n{'='*55}")
    print(f" {task_name}: 5-Fold Stratified CV")
    print(f"{'='*55}")
    print(df_m.to_string(index=False))
    print(f"\n  평균  Recall={df_m['recall'].mean():.3f}  "
          f"Precision={df_m['precision'].mean():.3f}  "
          f"F1={df_m['f1'].mean():.3f}  "
          f"AUC={df_m['auc'].mean():.3f}")
    print(f"  표준편차  Recall={df_m['recall'].std():.3f}  F1={df_m['f1'].std():.3f}")
    print(f"\nOOF Confusion Matrix:")
    print(confusion_matrix(y, oof_preds))
    print(f"\n{classification_report(y, oof_preds, zero_division=0)}")

    # 폴드 평균과 별도로 OOF 전체 집계 지표 출력 (evaluate.py와 동일 기준)
    oof_auc = roc_auc_score(y, oof_proba) if len(np.unique(y)) > 1 else np.nan
    print(f"OOF 전체 집계  "
          f"Recall={recall_score(y, oof_preds, zero_division=0):.3f}  "
          f"F1={f1_score(y, oof_preds, zero_division=0):.3f}  "
          f"AUC={oof_auc:.3f}")

    return df_m, oof_preds, oof_proba


def plot_cm(y_true, y_pred, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"저장: {save_path.name}")


# %%
# Task A 로드
df_a   = pd.read_csv(PROCESSED_DIR / "dataset_task_a.csv")
feat_a = [c for c in df_a.columns if c not in META_COLS + ["task_a"]]
X_a    = df_a[feat_a]
y_a    = df_a["task_a"].values
spw_a  = (y_a == 0).sum() / (y_a == 1).sum()

print(f"Task A  샘플={len(y_a)}  피처={len(feat_a)}")
print(f"  양품(0)={(y_a==0).sum()}  불량(1)={(y_a==1).sum()}  scale_pos_weight={spw_a:.3f}")

# %%
# Task A: CV
metrics_a, oof_pred_a, oof_proba_a = run_cv(X_a, y_a, spw_a, "Task A (가공불량)")

# %%
# Task A: Confusion Matrix 시각화
plot_cm(y_a, oof_pred_a,
        labels=["양품(0)", "불량(1)"],
        title="Task A - XGBoost Confusion Matrix (OOF 5-Fold)",
        save_path=FIG_DIR / "10_cm_xgb_task_a.png")

# %%
# Task A: 전체 데이터로 최종 모델 학습 + 저장
final_a = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw_a)
final_a.fit(X_a, y_a)
final_a.save_model(MODELS_DIR / "xgb_task_a.json")

# SHAP 분석에서 쓸 피처명 저장
(MODELS_DIR / "xgb_task_a_features.json").write_text(
    json.dumps(feat_a, ensure_ascii=False, indent=2)
)
print(f"\nTask A 최종 모델 저장: models/xgb_task_a.json")

# %%
# Task B 로드
df_b   = pd.read_csv(PROCESSED_DIR / "dataset_task_b.csv")
feat_b = [c for c in df_b.columns if c not in META_COLS + ["task_b"]]
X_b    = df_b[feat_b]
y_b    = df_b["task_b"].values
spw_b  = (y_b == 0).sum() / (y_b == 1).sum()

print(f"\nTask B  샘플={len(y_b)}  피처={len(feat_b)}")
print(f"  unworn(0)={(y_b==0).sum()}  worn(1)={(y_b==1).sum()}  scale_pos_weight={spw_b:.3f}")

# %%
# Task B: CV
metrics_b, oof_pred_b, oof_proba_b = run_cv(X_b, y_b, spw_b, "Task B (공구마모)")

# %%
# Task B: Confusion Matrix 시각화
plot_cm(y_b, oof_pred_b,
        labels=["unworn(0)", "worn(1)"],
        title="Task B - XGBoost Confusion Matrix (OOF 5-Fold)",
        save_path=FIG_DIR / "11_cm_xgb_task_b.png")

# %%
# Task B: 전체 데이터로 최종 모델 학습 + 저장
final_b = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw_b)
final_b.fit(X_b, y_b)
final_b.save_model(MODELS_DIR / "xgb_task_b.json")

(MODELS_DIR / "xgb_task_b_features.json").write_text(
    json.dumps(feat_b, ensure_ascii=False, indent=2)
)
print(f"Task B 최종 모델 저장: models/xgb_task_b.json")

# %%
# OOF 예측 저장 (evaluate.py 에서 DNN 결과와 비교용)
oof_a = df_a[META_COLS + ["task_a"]].copy()
oof_a["xgb_pred"]  = oof_pred_a
oof_a["xgb_proba"] = oof_proba_a
oof_a.to_csv(PROCESSED_DIR / "oof_xgb_task_a.csv", index=False)

oof_b = df_b[META_COLS + ["task_b"]].copy()
oof_b["xgb_pred"]  = oof_pred_b
oof_b["xgb_proba"] = oof_proba_b
oof_b.to_csv(PROCESSED_DIR / "oof_xgb_task_b.csv", index=False)

# %%
# 성능 비교 요약
print("\n" + "="*55)
print(" XGBoost 성능 요약 (5-Fold CV 평균)")
print("="*55)
summary = pd.DataFrame({
    "Task":      ["Task A (가공불량)", "Task B (공구마모)"],
    "Recall":    [metrics_a["recall"].mean(),    metrics_b["recall"].mean()],
    "Precision": [metrics_a["precision"].mean(), metrics_b["precision"].mean()],
    "F1":        [metrics_a["f1"].mean(),        metrics_b["f1"].mean()],
    "AUC":       [metrics_a["auc"].mean(),       metrics_b["auc"].mean()],
}).round(3)
print(summary.to_string(index=False))
