import shap
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models"
FIG_DIR       = Path(__file__).parent.parent / "reports" / "figures"

plt.rcParams["font.family"]        = ["NanumSquare", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"]         = 130

META_COLS = ["No", "feedrate", "clamp_pressure"]


def shap_values_2d(explainer, X):
    """binary XGBoost는 log-odds 기준 단일 배열 반환.
    혹시 클래스별 리스트면 positive class(index 1)만 취함."""
    vals = explainer.shap_values(X)
    if isinstance(vals, list):
        vals = vals[1]
    return vals


# %%
# Task A: 데이터 + 모델 로드
df_a   = pd.read_csv(PROCESSED_DIR / "dataset_task_a.csv")
feat_a = [c for c in df_a.columns if c not in META_COLS + ["task_a"]]
X_a    = df_a[feat_a]
y_a    = df_a["task_a"].values

model_a = xgb.XGBClassifier()
model_a.load_model(MODELS_DIR / "xgb_task_a.json")
print(f"Task A 로드 완료: {X_a.shape[0]}샘플 × {X_a.shape[1]}피처")

# %%
# Task A: SHAP 계산
explainer_a = shap.TreeExplainer(model_a)
shap_vals_a = shap_values_2d(explainer_a, X_a)
base_a      = float(explainer_a.expected_value)
print(f"Task A SHAP 완료 / shape: {shap_vals_a.shape} / base: {base_a:.4f}")

# %%
# Task A: Summary plot (beeswarm) — 각 피처가 불량 확률을 올리는지 내리는지
shap.summary_plot(shap_vals_a, X_a, max_display=15, show=False)
plt.title("Task A (가공불량): SHAP Summary — 상위 15 피처", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "12_shap_summary_task_a.png", bbox_inches="tight")
plt.close()
print("저장: 12_shap_summary_task_a.png")

# %%
# Task A: Bar plot — mean |SHAP| 중요도 순위
shap.summary_plot(shap_vals_a, X_a, plot_type="bar", max_display=15, show=False)
plt.title("Task A (가공불량): 피처 중요도 (mean |SHAP|)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "13_shap_bar_task_a.png", bbox_inches="tight")
plt.close()
print("저장: 13_shap_bar_task_a.png")

# %%
# Task A: FN 케이스 Waterfall — 불량인데 양품으로 오판한 이유 분석
oof_a = pd.read_csv(PROCESSED_DIR / "oof_xgb_task_a.csv")
fn_a  = oof_a[(oof_a["task_a"] == 1) & (oof_a["xgb_pred"] == 0)].index.tolist()
print(f"\nTask A FN (불량→양품 오판): {len(fn_a)}개  인덱스={fn_a}")

for i, idx in enumerate(fn_a):
    exp_i = shap.Explanation(
        values        = shap_vals_a[idx],
        base_values   = base_a,
        data          = X_a.iloc[idx].values,
        feature_names = feat_a,
    )
    shap.plots.waterfall(exp_i, max_display=12, show=False)
    no = int(df_a.iloc[idx]["No"])
    plt.title(f"Task A FN: 실험 No.{no} (실제=불량, 예측=양품)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"14_shap_waterfall_fn_task_a_{i+1}.png", bbox_inches="tight")
    plt.close()
    print(f"저장: 14_shap_waterfall_fn_task_a_{i+1}.png  (실험 No.{no})")

# %%
# Task A: 상위 15 피처 출력
imp_a = (pd.DataFrame({"feature": feat_a,
                        "mean_shap": np.abs(shap_vals_a).mean(axis=0)})
           .sort_values("mean_shap", ascending=False)
           .head(15).reset_index(drop=True))
print("\n=== Task A: 상위 15 피처 ===")
print(imp_a.to_string())

# %%
# Task B: 데이터 + 모델 로드
df_b   = pd.read_csv(PROCESSED_DIR / "dataset_task_b.csv")
feat_b = [c for c in df_b.columns if c not in META_COLS + ["task_b"]]
X_b    = df_b[feat_b]
y_b    = df_b["task_b"].values

model_b = xgb.XGBClassifier()
model_b.load_model(MODELS_DIR / "xgb_task_b.json")
print(f"\nTask B 로드 완료: {X_b.shape[0]}샘플 × {X_b.shape[1]}피처")

# %%
# Task B: SHAP 계산
explainer_b = shap.TreeExplainer(model_b)
shap_vals_b = shap_values_2d(explainer_b, X_b)
base_b      = float(explainer_b.expected_value)
print(f"Task B SHAP 완료 / shape: {shap_vals_b.shape} / base: {base_b:.4f}")

# %%
# Task B: Summary plot
shap.summary_plot(shap_vals_b, X_b, max_display=15, show=False)
plt.title("Task B (공구마모): SHAP Summary — 상위 15 피처", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "15_shap_summary_task_b.png", bbox_inches="tight")
plt.close()
print("저장: 15_shap_summary_task_b.png")

# %%
# Task B: Bar plot
shap.summary_plot(shap_vals_b, X_b, plot_type="bar", max_display=15, show=False)
plt.title("Task B (공구마모): 피처 중요도 (mean |SHAP|)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "16_shap_bar_task_b.png", bbox_inches="tight")
plt.close()
print("저장: 16_shap_bar_task_b.png")

# %%
# Task B: FN 케이스 Waterfall
oof_b = pd.read_csv(PROCESSED_DIR / "oof_xgb_task_b.csv")
fn_b  = oof_b[(oof_b["task_b"] == 1) & (oof_b["xgb_pred"] == 0)].index.tolist()
print(f"\nTask B FN (worn→unworn 오판): {len(fn_b)}개  인덱스={fn_b}")

for i, idx in enumerate(fn_b[:3]):
    exp_i = shap.Explanation(
        values        = shap_vals_b[idx],
        base_values   = base_b,
        data          = X_b.iloc[idx].values,
        feature_names = feat_b,
    )
    shap.plots.waterfall(exp_i, max_display=12, show=False)
    no = int(df_b.iloc[idx]["No"])
    plt.title(f"Task B FN: 실험 No.{no} (실제=worn, 예측=unworn)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"17_shap_waterfall_fn_task_b_{i+1}.png", bbox_inches="tight")
    plt.close()
    print(f"저장: 17_shap_waterfall_fn_task_b_{i+1}.png  (실험 No.{no})")

# %%
# Task B: 상위 15 피처 출력
imp_b = (pd.DataFrame({"feature": feat_b,
                        "mean_shap": np.abs(shap_vals_b).mean(axis=0)})
           .sort_values("mean_shap", ascending=False)
           .head(15).reset_index(drop=True))
print("\n=== Task B: 상위 15 피처 ===")
print(imp_b.to_string())

print("\n=== SHAP 분석 완료 ===")
