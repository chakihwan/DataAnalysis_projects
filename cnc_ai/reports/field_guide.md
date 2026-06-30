# CNC 가공 AI 현장 적용 가이드

작성일: 2026-06-26
대상 독자: 제조 현장 엔지니어 / MES 담당자 / 데이터 엔지니어
참조: `reports/modeling_report.md`, `reports/eda_report.md`

---

## 1. 이 모델이 하는 일

이 시스템은 CNC 가공 완료 후 센서 데이터를 분석해 두 가지를 판단한다.

| | Task A | Task B |
|--|--|--|
| **판단 내용** | 이 가공품이 불량인가? | 이 공구가 마모됐는가? |
| **입력** | 가공 중 수집된 센서 시계열 | 동일 |
| **출력** | 양품(0) / 불량(1) | unworn(0) / worn(1) |
| **권고 모델** | XGBoost (일반) / DNN+threshold=0.40 (불량 유출 최소화) | XGBoost |
| **핵심 성능** | Recall 0.750 (XGBoost) / Recall 0.917 (DNN, th=0.40) | Recall 0.571 |

**현장 언어로 번역하면:**
- Task A (일반 운영): 불량 10건 중 약 7~8건을 자동으로 걸러냄. 나머지 2~3건은 기존 육안검사 유지 필요
- Task A (불량 유출 최소화): DNN + threshold=0.40 사용 시 불량 12건 중 11건 탐지(Recall=0.917). 양품 중 약 21%를 불량으로 오판(FP)하므로 해당 물량에 대한 2차 검사 공정 필요
- Task B: 마모 10건 중 약 5~6건 탐지. 신뢰도가 낮아 보조 지표로 활용 권고

---

## 2. 시스템 구성 요구사항

### 2-1. 필수 센서 목록

아래 센서 신호가 있어야 피처 추출이 가능하다.
신호 이름은 기계마다 다를 수 있으므로 **컬럼 매핑 테이블** (섹션 3) 참조.

| 센서 | 측 | 중요도 |
|--|--|--|
| OutputCurrent | X, Y, Z, S | 필수 |
| OutputVoltage | X, Y, S | 필수 |
| OutputPower | X, Y, S | 필수 |
| ActualVelocity | X, Y, Z, S | 필수 (Task A 핵심) |
| SystemInertia | S | 권고 |

> **Z축 센서 참고:** 이번 데이터에서 Z축은 판별력이 없었으나, 기계 구성에 따라 다를 수 있으므로 수집은 유지하는 것이 좋다.

### 2-2. 공정 단계(Stage) 마킹 필수

센서 시계열에서 **절삭 구간(Layer)** 을 구분할 수 있어야 한다. 이 데이터에서는 `machining_process` 컬럼으로 구분했다.

```
필요한 구간 레이블:
  - Layer 1 Up / Layer 1 Down
  - Layer 2 Up / Layer 2 Down
  - Layer 3 Up / Layer 3 Down
```

절삭 구간 이외(Prep, Repositioning 등)는 피처 추출에서 노이즈로 작용하므로 반드시 분리해야 한다. 기계에서 공정 단계를 자동으로 마킹하지 않는 경우, G-code 실행 이벤트 로그나 스핀들 ON/OFF 신호를 대리 지표로 활용할 수 있다.

### 2-3. 데이터 수집 주기

- 원본 데이터는 약 100ms 주기 수집
- 피처 추출은 구간 집계이므로 수집 주기가 달라도 무방 (단, 너무 성글면 통계량 불안정)
- 최소 50ms 주기 이상 권고

---

## 3. 컬럼 매핑 테이블

FANUC, SIEMENS, Mazak 등 제조사마다 신호 이름이 다르다. 아래 테이블을 현장 신호명으로 채워 `preprocessing.py`의 `SENSOR_COLS`와 매핑한다.

| 모델 내부 이름 | 물리적 의미 | FANUC 예시 | 현장 실제 신호명 (기입 필요) |
|--|--|--|--|
| `X_OutputCurrent` | X축 서보 전류 | PMC R3000 | ________ |
| `X_OutputVoltage` | X축 서보 전압 | PMC R3002 | ________ |
| `X_OutputPower` | X축 서보 파워 | PMC R3004 | ________ |
| `X_ActualVelocity` | X축 실제 이송속도 | Feedrate actual | ________ |
| `Y_OutputCurrent` | Y축 서보 전류 | — | ________ |
| `Y_OutputVoltage` | Y축 서보 전압 | — | ________ |
| `Y_OutputPower` | Y축 서보 파워 | — | ________ |
| `Y_ActualVelocity` | Y축 실제 이송속도 | — | ________ |
| `Z_OutputCurrent` | Z축 서보 전류 | — | ________ |
| `Z_OutputVoltage` | Z축 서보 전압 | — | ________ |
| `Z_OutputPower` | Z축 서보 파워 | — | ________ |
| `Z_ActualVelocity` | Z축 실제 이송속도 | — | ________ |
| `S_OutputCurrent` | 스핀들 전류 | — | ________ |
| `S_OutputVoltage` | 스핀들 전압 | — | ________ |
| `S_OutputPower` | 스핀들 파워 | — | ________ |
| `S_ActualVelocity` | 스핀들 실제 회전수(RPM) | Spindle actual speed | ________ |
| `S_SystemInertia` | 스핀들 관성값 | — | ________ |
| `machining_process` | 공정 단계 레이블 | G-code block tag | ________ |

---

## 4. 현장 데이터로 재학습하는 방법

### 왜 재학습이 필요한가

이 모델은 University of Michigan SMART Lab의 통제 실험(feedrate 5단계, pressure 3단계, 알루미늄 소재)으로 학습됐다. 현장에서 소재가 다르거나, 가공 속도 범위가 다르거나, 공구 종류가 다르면 학습 도메인과 달라져 예측력이 저하된다.

**현장 데이터로 재학습하면 모델이 해당 공정에 최적화된다.**

### 재학습 절차

**Step 1: 현장 데이터 수집 및 레이블링**

```
수집 목표:
- 최소 100건 이상의 가공 완료 사례
- 양품 : 불량 = 가능한 한 실제 비율 반영
  (실제 현장의 불량률이 5%라면 100건 중 5건 이상 불량 포함)
- Task B: 마모 공구 / 신품 공구 각각 최소 30건 이상

레이블링 방법:
- Task A: 기존 품질 검사 결과 (치수 측정, 표면 검사) 기록 필수
- Task B: 공구 교체 이력 기록 + 교체 전 마지막 n건 = worn 레이블
```

**Step 2: 데이터 전처리**

현장 데이터를 `data/raw/` 에 넣고 컬럼명을 섹션 3의 매핑 테이블에 따라 맞춘 뒤:

```bash
python src/preprocessing.py
```

피처 추출 로직은 동일하게 작동한다. 컬럼명이 달라도 매핑 테이블로 처리 가능.

**Step 3: 재학습**

```bash
python src/train_xgboost.py
```

샘플 수가 늘면 하이퍼파라미터 재조정을 검토한다:

| 샘플 수 | 권고 변경 |
|--|--|
| 25~99건 | 기존 파라미터 유지 |
| 100~500건 | `max_depth=4~5`, `reg_alpha=0.5` 완화 검토 |
| 500건 이상 | Cross-validation으로 최적화, DNN 재검토 가능 |

**Step 4: 검증**

```bash
python src/evaluate.py
```

Recall이 현장에서 요구하는 수준(예: "불량 미통과율 90% 이상")을 충족하는지 확인.

---

## 5. 실시간 추론 파이프라인

### 추론 흐름

```
가공 완료
    ↓
센서 시계열 수집 (machining_process 레이블 포함)
    ↓
src/preprocessing.py 로직으로 집계 → 1×177 피처 벡터
    ↓
xgb_task_a.json 로드 → 불량 확률 출력
xgb_task_b.json 로드 → 마모 확률 출력
    ↓
threshold 적용 → 판정 (양품/불량, unworn/worn)
    ↓
MES / 알람 시스템 연동
```

### 추론 코드 예시 — XGBoost (일반 운영)

```python
import xgboost as xgb
import pandas as pd
import json

# 모델 로드 (서버 시작 시 1회)
model_a = xgb.XGBClassifier()
model_a.load_model("models/xgb_task_a.json")
feat_a = json.loads(open("models/xgb_task_a_features.json").read())

# 가공 완료 후 피처 추출 (preprocessing.py 로직 활용)
# feature_vector: 1행 × 177컬럼 DataFrame
feature_vector = extract_features(sensor_data)  # 피처 추출 함수

# 피처 순서 보장 (반드시 필요)
X = feature_vector[feat_a]

# 불량 확률
proba = model_a.predict_proba(X)[:, 1][0]

# 판정 (threshold 조정 가능)
threshold = 0.5
result = "불량" if proba >= threshold else "양품"
print(f"불량 확률: {proba:.3f} → {result}")
```

### 추론 코드 예시 — DNN (불량 절대 불통과 시나리오, threshold=0.33)

```python
import torch
import torch.nn as nn
import numpy as np
import json

# DNN 아키텍처 (train_dnn.py와 동일하게 정의)
class DefectDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32),        nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# 모델 로드 (서버 시작 시 1회)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("models/dnn_task_a.pt", map_location=device)
model = DefectDNN(ckpt["input_dim"])
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

scaler_mean  = ckpt["scaler_mean"]
scaler_scale = ckpt["scaler_scale"]

# 가공 완료 후 피처 추출
feature_vector = extract_features(sensor_data)   # 1행 × 177컬럼 numpy array
X_scaled = (feature_vector - scaler_mean) / scaler_scale
X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# 불량 확률
with torch.no_grad():
    logit = model(X_t).item()
proba = 1 / (1 + np.exp(-logit))

# 판정 — Recall 1.000 달성 threshold
THRESHOLD = 0.40
result = "불량(재검사)" if proba >= THRESHOLD else "양품"
print(f"불량 확률: {proba:.3f} → {result}")
```

> **주의:** DNN 추론 시 반드시 모델 파일에 저장된 scaler 파라미터(mean, scale)를 사용해야 한다. 학습 때와 다른 스케일링을 적용하면 예측이 무의미해진다.

### Threshold 조정 가이드

| 모델 | threshold | Recall | Precision | F1 | 적합한 상황 |
|--|--|--|--|--|--|
| XGBoost | 0.50 (기본) | 0.750 | 0.750 | 0.750 | 일반 운영, 균형 중시 |
| XGBoost | 0.48 | 0.833 | 0.769 | 0.800 | Recall 강화 (F1 최대) |
| DNN | 0.40 | 0.917 | 0.786 | 0.846 | 불량 유출 최소화 (오탐 21% 허용) |

ROC Curve (그림 21)와 Threshold 분석 (그림 24~25)를 보고 현장 요구에 맞는 조합을 선택한다.

---

## 6. 모델이 주목하는 신호 - 현장 해석

SHAP 분석에서 도출한 핵심 물리 신호:

### Task A (가공불량) 핵심 신호

**S_ActualVelocity (스핀들 실제 회전수)** — 가장 중요한 단일 신호

> 불량 가공 중에는 스핀들 속도가 지시값보다 떨어지는 경향이 있다.
> 가공 저항이 커지거나(과부하), 진동으로 인한 속도 변동이 생기면 불량으로 이어진다.

**현장 체크포인트:**
- 가공 중 스핀들 RPM이 설정값 대비 10% 이상 낮아지는 구간이 있는가?
- 스핀들 RPM 변동(std)이 평소보다 큰가?

### Task B (공구마모) 핵심 신호

**Z_ActualVelocity_cut_std (절삭 중 Z축 이송속도 변동성)**

> 공구가 마모될수록 절삭력이 불안정해지며, 이것이 Z축 이송속도의 진동으로 나타난다.
> 단, 이번 모델의 Task B 성능이 낮은 만큼, 이 신호 하나만으로 판단하지 말 것.

**현장 체크포인트:**
- 동일 조건 가공에서 Z축 이송이 최근 들어 불안정해졌는가?
- 공구 교체 주기가 도래했는가 (누적 가공 시간 기준)?

---

## 7. 한계 및 향후 개선 방향

### 현재 모델의 한계

| 한계 | 원인 | 영향 |
|--|--|--|
| 학습 샘플 25건 | 실험 데이터 특성 | 과적합 위험, 일반화 불확실 |
| Task B Recall 0.571 | 공구마모 신호 약함 | 마모 탐지 신뢰도 낮음 |
| 단일 소재(알루미늄) | 실험 설계 한계 | 다른 소재 적용 불가 |
| 단일 공구 종류 | 동일 | 공구 교체 시 재학습 필요 |

### 향후 개선 방향

**단기 (현장 데이터 수집 단계):**
- 현장 실제 데이터 100건 이상 수집 → 재학습
- 불량 유형별 레이블 세분화 (치수불량 vs 표면불량 등)

**중기 (Task B 개선):**
- 누적 가공 시간/거리 피처 추가 (공구 수명 모델링)
- 레이어별 절삭력 변화율(기울기) 피처 추가
- LSTM 기반 시계열 직접 학습 검토

**장기 (시스템 완성):**
- 실시간 이상 감지 (가공 중간에 알람)
- 예지보전 시스템 연동 (공구 교체 시점 예측)
- 여러 기계에서 수집한 데이터 통합 학습 (도메인 적응)

---

## 8. 빠른 시작 체크리스트

```
[ ] 센서 컬럼명 매핑 테이블 작성 (섹션 3)
[ ] 공정 단계(Layer) 마킹 방법 확인
[ ] 현장 데이터 100건 수집 계획 수립
[ ] 레이블링 기준 정의 (Task A: 품질 검사 기준, Task B: 공구 교체 이력)
[ ] Python 환경 구성 (requirements.txt 참조)
[ ] preprocessing.py 컬럼 매핑 적용 후 실행 테스트
[ ] train_xgboost.py 재학습
[ ] evaluate.py 성능 검증
[ ] threshold 결정 (현장 Recall 요구 기준)
[ ] MES/알람 연동 API 개발
```
