# CNC 가공 공정 AI 분석 프로젝트

## 상위 규칙 참조
공통 개발 규칙은 상위 폴더의 CLAUDE.md 참조
D:\WorkSpace\DataAnalysis_projects\CLAUDE.md

## 프로젝트 구조
cnc_ai/
├── CLAUDE.md
├── README_data.md             ← 데이터 탐색 후 자동 생성
├── data/
│   ├── raw/                   ← 원본 데이터 (건드리지 않음)
│   │   ├── train.csv
│   │   └── experiment_01~25.csv
│   └── processed/             ← 전처리 후 자동 생성
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train_xgboost.py
│   ├── train_dnn.py
│   ├── shap_analysis.py
│   └── evaluate.py
├── reports/
│   └── figures/
└── models/

## 프로젝트 목적
KAMP CNC 머신 공개 데이터를 활용하여
제조 현장에 이식 가능한 AI 파이프라인 구축.
단순 정확도 달성이 아닌, 실제 현장 적용을
고려한 방법론 검증이 목표.

## 데이터 파일
- train.csv: 실험별 메타정보
- experiment_01.csv ~ experiment_25.csv: 센서 시계열 데이터
- 상세 구조는 데이터 탐색 후 README_data.md 에 문서화

## 태스크 정의

### Task A: 가공불량 예측 (이진분류)
타겟: 양품(0) / 불량(1)
- 양품 조건: machining_finalized==yes AND passed_visual_inspection==yes
- 불량 조건: 그 외 전부

### Task B: 공구마모 분류 (이진분류)
타겟: unworn(0) / worn(1)
- tool_condition 컬럼 직접 사용

## 모델링 원칙
- XGBoost: 기준 모델 (먼저 학습)
- DNN (PyTorch): 비교 모델
- 두 모델을 동일한 평가 기준으로 비교
- 주요 지표: Recall 최우선, 보조: F1-Score / AUC / Confusion Matrix

## 현장 이식 설계 원칙
- 피처를 물리적 의미 기반으로 그룹화
  → 컬럼명이 달라도 그룹 매핑으로 재사용 가능하게 설계
- 마지막에 현장 적용 가이드 문서 생성
  → 실제 FANUC/SIEMENS 머신 적용 시
     컬럼 매핑 테이블 + 재학습 절차 포함

## 분석 문서화 원칙

### 코드와 인사이트 분리
- **코드** (`src/*.py`): 시각화, 통계 계산, figure 저장만 담당
- **분석 내러티브** (`reports/*.md`): 도메인 해석, 가설, 인사이트를 사람이 작성
- 코드에 인사이트를 주석으로 끼워넣지 않음 → 둘 다 어중간해지기 때문

### 워크플로우
1. `src/*.py` 실행 → figures / 통계 생성
2. 결과를 보며 대화로 인사이트 도출
3. 도출된 내용을 `reports/*.md`에 문서화
- 인사이트는 자동생성 불가 → 도메인 지식 기반 대화에서 나옴

### 분석 문서 목록
- `reports/eda_report.md`: EDA 인사이트 (분포 해석, 가설, 도메인 의미)
- `reports/modeling_report.md`: 모델 선택 근거, 실험 결과 해석
- `reports/field_guide.md`: 현장 적용 가이드 (최종 산출물)

## 작업 순서
1. data/raw/ 탐색 → README_data.md 자동 생성 ✅
2. EDA (eda.py) → 결과 논의 → eda_report.md 작성
3. 전처리 (preprocessing.py)
4. Task A 모델링: XGBoost → DNN → 비교 (train_xgboost.py, train_dnn.py)
5. Task B 모델링: XGBoost → DNN → 비교
6. SHAP 해석 (shap_analysis.py)
7. 두 태스크 종합 평가 (evaluate.py)
8. 현장 적용 가이드 문서 생성 (field_guide.md)

## 결과물 목록
- README_data.md (데이터 구조 문서)
- EDA 시각화 (분포, 박스플롯, 히트맵, 이상치)
- 양품 vs 불량 분포 비교
- XGBoost vs DNN 성능 비교표 (Task A, B 각각)
- SHAP Feature Importance 시각화
- Confusion Matrix / ROC Curve
- 현장 적용 가이드 문서
