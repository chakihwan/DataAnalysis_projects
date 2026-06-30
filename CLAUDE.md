# DataAnalysis Projects - 공통 개발 규칙

## 프로젝트 구조
```
D:\WorkSpace\DataAnalysis_projects\
├── CLAUDE.md                  ← 이 파일 (공통 규칙)
├── Dockerfile                 ← 공용 컨테이너
├── docker-compose.yml
├── requirements.txt
├── .devcontainer/
│   └── devcontainer.json
│
├── cnc_ai/                    ← 프로젝트별 폴더
│   └── CLAUDE.md              ← 프로젝트 전용 규칙
│
└── (추후 프로젝트 폴더 추가)
```

## 개발 환경

### 로컬 환경
- OS: Windows 11
- GPU: NVIDIA GeForce RTX 5060 (8GB VRAM)
- CUDA: 13.0
- Docker Desktop (WSL2 백엔드)
- VSCode + Dev Containers 확장

### 컨테이너 환경
- 베이스 이미지: pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
- 공용 컨테이너 (모든 프로젝트 공유)
- GPU 패스스루 활성화
- 모든 프로젝트 폴더를 /workspace 에 마운트

### VSCode 작업 방식
- .py 파일에서 `# %%` 마커로 셀 단위 실행 (Shift+Enter)
- Jupyter Notebook(.ipynb) 사용 금지
  → Git 관리 불편, Claude Code가 읽기 어려움

## 공통 패키지
```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
ipykernel
```

## 코드 규칙

### 기본 원칙
- 재현성: random_state=42 고정
- 함수 단위로 모듈화 (한 함수 = 한 역할)
- 결과물(그래프 등)은 각 프로젝트의 reports/figures/ 에 저장
- 원본 데이터(data/raw/)는 절대 수정하지 않음
- 전처리 결과는 data/processed/ 에 저장

### 딥러닝 프레임워크
- PyTorch 사용 (TensorFlow/Keras 사용 금지)
- 이유 1: RTX 5060 Blackwell 아키텍처 호환성
- 이유 2: 채용시장 대세 (LLM/생성AI 전부 PyTorch 기반)
- TensorFlow 기반 코드 요청이 와도 PyTorch로 구현할 것

### 클래스 불균형 대응 전략
제조 데이터 특성상 양품:불량 비율이 심하게 불균형할 수 있음
1. 평가지표는 Accuracy 대신 F1-Score / Recall 우선 사용
2. 클래스 가중치 부여 우선 적용 (scale_pos_weight 등)
3. 필요시 SMOTE 오버샘플링 적용
4. Recall(불량을 불량이라 맞춘 비율)을 최우선 지표로
   → 불량을 양품으로 통과시키는 게 제일 위험하기 때문

### Feature Importance
- SHAP 사용 권장
- 이유: 모델 종류(XGBoost, DNN 등)에 관계없이
  동일한 방식으로 비교 가능하고
  "왜 이 샘플이 불량인가" 현장 설명이 가능함
- Layer 1 가중치 방식 사용 금지
  → 뒤 레이어에서 뒤집힐 수 있어 부정확
