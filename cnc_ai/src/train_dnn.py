import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (recall_score, precision_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models"
FIG_DIR       = Path(__file__).parent.parent / "reports" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"]        = "NanumSquare"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"]         = 130

META_COLS    = ["No", "feedrate", "clamp_pressure"]
RANDOM_STATE = 42
EPOCHS       = 300
PATIENCE     = 30   # EarlyStopping: val_loss 개선 없는 epoch 수
LR           = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT      = 0.4

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# %%
# DNN 아키텍처 정의
# 25샘플로 과적합 방지 → 얕은 네트워크 + Dropout + BatchNorm + WeightDecay
class DefectDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # BatchNorm은 소수 샘플(20개)에서 수치 불안정 → LayerNorm 사용
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# %%
# 공통 CV 함수
def run_cv(X_np, y_np, pos_weight_val, task_name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = []
    oof_preds = np.zeros(len(y_np), dtype=int)
    oof_proba = np.zeros(len(y_np))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_np, y_np), 1):
        X_tr_raw, X_val_raw = X_np[tr_idx], X_np[val_idx]
        y_tr, y_val         = y_np[tr_idx],  y_np[val_idx]

        # NaN 처리: 일부 layer별 통계량이 NaN일 수 있음 (절삭 구간 없는 경우)
        X_tr_raw  = np.nan_to_num(X_tr_raw, nan=0.0)
        X_val_raw = np.nan_to_num(X_val_raw, nan=0.0)

        # 피처 스케일링 (DNN은 스케일에 민감, fold 내 fit)
        scaler = StandardScaler()
        X_tr  = scaler.fit_transform(X_tr_raw)
        X_val = scaler.transform(X_val_raw)

        # Tensor 변환
        X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32).to(device)
        y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

        # 모델 / 손실함수 / 옵티마이저
        model     = DefectDNN(X_tr.shape[1]).to(device)
        pw        = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=EPOCHS)

        # 학습 (full-batch) + EarlyStopping (val_loss 기준)
        best_val_loss  = float('inf')
        patience_count = 0
        best_state     = None

        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            logits = model(X_tr_t)
            loss   = criterion(logits, y_tr_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # val_loss 체크
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                best_state     = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    break

        # val_loss 최소 시점의 가중치로 복원
        if best_state is not None:
            model.load_state_dict(best_state)

        # 검증
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t).cpu().numpy()
        proba = 1 / (1 + np.exp(-logits_val))   # sigmoid
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
    print(f" {task_name}: DNN 5-Fold Stratified CV")
    print(f"{'='*55}")
    print(df_m.to_string(index=False))
    print(f"\n  평균  Recall={df_m['recall'].mean():.3f}  "
          f"Precision={df_m['precision'].mean():.3f}  "
          f"F1={df_m['f1'].mean():.3f}  "
          f"AUC={df_m['auc'].mean():.3f}")
    print(f"  표준편차  Recall={df_m['recall'].std():.3f}  F1={df_m['f1'].std():.3f}")
    print(f"\nOOF Confusion Matrix:")
    print(confusion_matrix(y_np, oof_preds))
    print(f"\n{classification_report(y_np, oof_preds, zero_division=0)}")

    return df_m, oof_preds, oof_proba


def train_final(X_np, y_np, pos_weight_val, save_path):
    """전체 데이터로 최종 모델 학습 + 저장"""
    X_np   = np.nan_to_num(X_np, nan=0.0)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_np)

    X_t = torch.tensor(X_sc, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_np, dtype=torch.float32).to(device)

    model     = DefectDNN(X_sc.shape[1]).to(device)
    pw        = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS)

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    torch.save({"model_state": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "input_dim": X_sc.shape[1]}, save_path)
    print(f"저장: {save_path.name}")
    return model, scaler


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
    plt.close()
    print(f"저장: {save_path.name}")


# %%
# Task A
df_a   = pd.read_csv(PROCESSED_DIR / "dataset_task_a.csv")
feat_a = [c for c in df_a.columns if c not in META_COLS + ["task_a"]]
X_a    = df_a[feat_a].values
y_a    = df_a["task_a"].values
spw_a  = (y_a == 0).sum() / (y_a == 1).sum()

print(f"\nTask A  샘플={len(y_a)}  피처={len(feat_a)}")
print(f"  양품(0)={(y_a==0).sum()}  불량(1)={(y_a==1).sum()}  pos_weight={spw_a:.3f}")

# %%
# Task A: CV
metrics_a, oof_pred_a, oof_proba_a = run_cv(X_a, y_a, spw_a, "Task A (가공불량)")

# %%
# Task A: Confusion Matrix
plot_cm(y_a, oof_pred_a,
        labels=["양품(0)", "불량(1)"],
        title="Task A - DNN Confusion Matrix (OOF 5-Fold)",
        save_path=FIG_DIR / "18_cm_dnn_task_a.png")

# %%
# Task A: 최종 모델 저장
train_final(X_a, y_a, spw_a, MODELS_DIR / "dnn_task_a.pt")

# %%
# Task A: OOF 저장
oof_a = df_a[META_COLS + ["task_a"]].copy()
oof_a["dnn_pred"]  = oof_pred_a
oof_a["dnn_proba"] = oof_proba_a
oof_a.to_csv(PROCESSED_DIR / "oof_dnn_task_a.csv", index=False)

# %%
# Task B
df_b   = pd.read_csv(PROCESSED_DIR / "dataset_task_b.csv")
feat_b = [c for c in df_b.columns if c not in META_COLS + ["task_b"]]
X_b    = df_b[feat_b].values
y_b    = df_b["task_b"].values
spw_b  = (y_b == 0).sum() / (y_b == 1).sum()

print(f"\nTask B  샘플={len(y_b)}  피처={len(feat_b)}")
print(f"  unworn(0)={(y_b==0).sum()}  worn(1)={(y_b==1).sum()}  pos_weight={spw_b:.3f}")

# %%
# Task B: CV
metrics_b, oof_pred_b, oof_proba_b = run_cv(X_b, y_b, spw_b, "Task B (공구마모)")

# %%
# Task B: Confusion Matrix
plot_cm(y_b, oof_pred_b,
        labels=["unworn(0)", "worn(1)"],
        title="Task B - DNN Confusion Matrix (OOF 5-Fold)",
        save_path=FIG_DIR / "19_cm_dnn_task_b.png")

# %%
# Task B: 최종 모델 저장
train_final(X_b, y_b, spw_b, MODELS_DIR / "dnn_task_b.pt")

# %%
# Task B: OOF 저장
oof_b = df_b[META_COLS + ["task_b"]].copy()
oof_b["dnn_pred"]  = oof_pred_b
oof_b["dnn_proba"] = oof_proba_b
oof_b.to_csv(PROCESSED_DIR / "oof_dnn_task_b.csv", index=False)

# %%
# 성능 비교 요약
print("\n" + "="*55)
print(" DNN 성능 요약 (5-Fold CV 평균)")
print("="*55)
summary = pd.DataFrame({
    "Task":      ["Task A (가공불량)", "Task B (공구마모)"],
    "Recall":    [metrics_a["recall"].mean(),    metrics_b["recall"].mean()],
    "Precision": [metrics_a["precision"].mean(), metrics_b["precision"].mean()],
    "F1":        [metrics_a["f1"].mean(),        metrics_b["f1"].mean()],
    "AUC":       [metrics_a["auc"].mean(),       metrics_b["auc"].mean()],
}).round(3)
print(summary.to_string(index=False))
