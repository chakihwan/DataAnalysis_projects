# CNC 가공 AI 파이프라인

KAMP 공개 데이터셋을 활용한 CNC 가공 공정 이상 감지 시스템.
단순 정확도 달성이 아닌 **실제 현장 이식을 고려한 방법론 검증**을 목표로 한다.

---

## 태스크 정의

| | Task A | Task B |
|--|--|--|
| **문제** | 가공불량 예측 | 공구마모 분류 |
| **타겟** | 양품(0) / 불량(1) | unworn(0) / worn(1) |
| **핵심 지표** | Recall (불량 미통과 최소화) | Recall |
| **권고 모델 (균형 운영)** | XGBoost (th=0.50) | XGBoost (th=0.50) |
| **권고 모델 (불량 유출 최소화)** | DNN (th=0.40, Recall=0.917) | — |

---

## 최종 성능 요약

### Task A — 가공불량

| 모델 | Threshold | Recall | Precision | F1 | AUC |
|--|--|--|--|--|--|
| XGBoost | 0.50 | 0.750 | 0.750 | 0.750 | 0.885 |
| XGBoost | 0.48 | 0.833 | 0.769 | 0.800 | — |
| DNN | 0.40 | **0.917** | 0.786 | **0.846** | **0.950** |

### Task B — 공구마모

| 모델 | Threshold | Recall | Precision | F1 | AUC |
|--|--|--|--|--|--|
| XGBoost | 0.50 | 0.571 | 0.571 | 0.571 | 0.714 |
| DNN | 0.50 | 0.643 | 0.643 | 0.643 | 0.767 |

> Task B는 학습 샘플 25건 한계로 Recall이 낮다. 현장 데이터 수집 후 재학습 권고.

---

## 주요 인사이트

**Task A 핵심 피처 (SHAP)**
- `S_ActualVelocity` — 불량 가공 중 스핀들 속도 저하
- `S_OutputPower`, `S_OutputVoltage` — 스핀들 출력 감소
- 불량 실험은 가공 중단으로 행 수 자체가 짧음 (`row_count` 유효)

**Task B 핵심 피처 (SHAP)**
- `Z_ActualVelocity_cut_std` — 마모 공구의 Z축 이송 불안정 진동 (1위)
- EDA 선형 상관으로는 Z축이 무신호처럼 보였으나, SHAP에서 비선형 패턴 확인

**DNN vs XGBoost**
- 샘플 25건 환경에서 XGBoost가 안정적
- DNN은 EarlyStopping(val_loss, patience=30) + GPU 학습으로 AUC Task A 0.830 → 0.950
- DNN의 높은 AUC를 활용해 threshold=0.40으로 내리면 Task A Recall 0.917 달성

---

## 프로젝트 구조

```
DataAnalysis_projects/
├── cnc_ai/
│   ├── data/
│   │   ├── raw/                  ← 원본 데이터 (수정 금지)
│   │   └── processed/            ← 전처리 결과 + OOF 예측
│   ├── models/
│   │   ├── xgb_task_a.json
│   │   ├── xgb_task_b.json
│   │   ├── dnn_task_a.pt         ← scaler 파라미터 내장
│   │   └── dnn_task_b.pt
│   ├── src/
│   │   ├── eda.py
│   │   ├── preprocessing.py
│   │   ├── train_xgboost.py
│   │   ├── train_dnn.py
│   │   ├── shap_analysis.py
│   │   ├── evaluate.py
│   │   └── threshold_analysis.py
│   └── reports/
│       ├── eda_report.md
│       ├── modeling_report.md
│       ├── field_guide.md        ← 현장 적용 가이드
│       └── figures/              ← 시각화 결과 (26개)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 실행 방법

### 컨테이너 환경 (권장)

```bash
docker-compose up -d
```

### 분석 순서

```bash
# 1. EDA
python src/eda.py

# 2. 전처리 (피처 추출)
python src/preprocessing.py

# 3. XGBoost 학습
python src/train_xgboost.py

# 4. DNN 학습
python src/train_dnn.py

# 5. SHAP 피처 중요도 분석
python src/shap_analysis.py

# 6. 모델 종합 평가
python src/evaluate.py

# 7. Threshold 최적화
python src/threshold_analysis.py
```

---

## 데이터

- 출처: [KAMP (한국 AI 제조 플랫폼)](https://www.kamp-ai.kr/) 공개 데이터셋
- 원본: University of Michigan SMART Lab CNC 통제 실험
- 규모: 알루미늄 가공 실험 25건 / 총 32,048행 센서 시계열
- 실험 조건: feedrate 5단계 (3~20 mm/min) × clamp_pressure 3단계 (2.5~4.0 bar)

---

## 현장 적용 가이드

`cnc_ai/reports/field_guide.md` 참조.
센서 컬럼 매핑 테이블, 재학습 절차, 실시간 추론 코드 예시(XGBoost / DNN), threshold 조정 가이드 포함.

---

## 개발 환경

- Python 3.10
- PyTorch 2.3.0 (컨테이너 베이스: cuda12.1-cudnn8-runtime)
- GPU: RTX 5060 Blackwell / CUDA 13.0 — PyTorch 2.3.0 Blackwell 미지원으로 **CPU 학습** (25샘플이라 수초 내 완료)
- XGBoost / SHAP / scikit-learn / pandas / matplotlib
