import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROJECT_DIR = Path(__file__).parent.parent

# %%
# train.csv 탐색
train = pd.read_csv(RAW_DIR / "train.csv")

print("=== train.csv ===")
print(f"Shape: {train.shape}")
print(f"Columns: {train.columns.tolist()}")
print()
print(train.to_string())
print()
print("결측치:")
print(train.isnull().sum())

# %%
# 타겟 레이블 분포 확인
print("\n=== Task A 타겟: 가공불량 ===")
train["defect"] = ~(
    (train["machining_finalized"] == "yes") &
    (train["passed_visual_inspection"] == "yes")
)
print(train["defect"].value_counts().rename({False: "양품(0)", True: "불량(1)"}))

print("\n=== Task B 타겟: 공구마모 ===")
print(train["tool_condition"].value_counts())

# %%
# experiment 파일 전체 탐색
exp_stats = []
first_exp = None

for i in range(1, 26):
    path = RAW_DIR / f"experiment_{i:02d}.csv"
    df = pd.read_csv(path)
    if first_exp is None:
        first_exp = df
    exp_stats.append({
        "no": i,
        "rows": len(df),
        "cols": len(df.columns),
        "machining_process_values": df["Machining_Process"].unique().tolist()
    })

print("\n=== experiment 파일별 행 수 ===")
for s in exp_stats:
    print(f"  experiment_{s['no']:02d}: {s['rows']}행")

print(f"\n총 센서 데이터 행 수: {sum(s['rows'] for s in exp_stats):,}")

# %%
# 센서 컬럼 그룹 분류
sensor_cols = [c for c in first_exp.columns if c not in ["Machining_Process"]]

axis_groups = {
    "X축 (테이블 X방향)": [c for c in sensor_cols if c.startswith("X_")],
    "Y축 (테이블 Y방향)": [c for c in sensor_cols if c.startswith("Y_")],
    "Z축 (수직방향)":     [c for c in sensor_cols if c.startswith("Z_")],
    "S축 (스핀들)":       [c for c in sensor_cols if c.startswith("S_")],
    "M (머신상태)":       [c for c in sensor_cols if c.startswith("M_")],
}

print("\n=== 센서 컬럼 그룹 ===")
for group, cols in axis_groups.items():
    print(f"  {group}: {cols}")

print(f"\n  Machining_Process: 공정 단계 레이블")
print(f"  값: {sorted(first_exp['Machining_Process'].unique())}")

# %%
# README_data.md 생성
total_rows = sum(s['rows'] for s in exp_stats)
row_min = min(s['rows'] for s in exp_stats)
row_max = max(s['rows'] for s in exp_stats)

train["defect"] = ~(
    (train["machining_finalized"] == "yes") &
    (train["passed_visual_inspection"] == "yes")
)
defect_counts = train["defect"].value_counts()
tool_counts = train["tool_condition"].value_counts()

readme = f"""# CNC 데이터 구조 문서 (README_data.md)

자동 생성 by explore_data.py

---

## 파일 구성

| 파일 | 설명 | Shape |
|------|------|-------|
| train.csv | 실험별 메타정보 및 레이블 | {train.shape[0]}행 × {train.shape[1]}열 |
| experiment_01~25.csv | 실험별 센서 시계열 | {row_min}~{row_max}행 × 48열 |

---

## train.csv 컬럼 설명

| 컬럼 | 타입 | 설명 | 값 |
|------|------|------|-----|
| No | int | 실험 번호 | 1~25 |
| material | str | 가공 재료 | aluminum (전체 동일) |
| feedrate | int | 이송 속도 (mm/min) | {sorted(train['feedrate'].unique())} |
| clamp_pressure | float | 클램프 압력 (bar) | {sorted(train['clamp_pressure'].unique())} |
| tool_condition | str | 공구 상태 | unworn / worn |
| machining_finalized | str | 가공 완료 여부 | yes / no |
| passed_visual_inspection | str | 육안 검사 통과 여부 | yes / no / NaN(미완료시) |

---

## 타겟 레이블 분포

### Task A: 가공불량 예측
- 양품(0): machining_finalized==yes AND passed_visual_inspection==yes
- 불량(1): 그 외

| 클래스 | 수 | 비율 |
|--------|-----|------|
| 양품(0) | {defect_counts.get(False, 0)} | {defect_counts.get(False, 0)/len(train)*100:.1f}% |
| 불량(1) | {defect_counts.get(True, 0)} | {defect_counts.get(True, 0)/len(train)*100:.1f}% |

### Task B: 공구마모 분류
- tool_condition 컬럼 직접 사용

| 클래스 | 수 | 비율 |
|--------|-----|------|
| unworn(0) | {tool_counts.get('unworn', 0)} | {tool_counts.get('unworn', 0)/len(train)*100:.1f}% |
| worn(1) | {tool_counts.get('worn', 0)} | {tool_counts.get('worn', 0)/len(train)*100:.1f}% |

---

## experiment_XX.csv 센서 컬럼 구조

총 {total_rows:,}행 (25개 실험 합산), 실험별 {row_min}~{row_max}행

### 축별 센서 그룹 (X/Y/Z/S 각 11개)

| 그룹 | 컬럼 |
|------|------|
| X축 (테이블 X방향) | {", ".join(axis_groups["X축 (테이블 X방향)"])} |
| Y축 (테이블 Y방향) | {", ".join(axis_groups["Y축 (테이블 Y방향)"])} |
| Z축 (수직방향) | {", ".join(axis_groups["Z축 (수직방향)"])} |
| S축 (스핀들) | {", ".join(axis_groups["S축 (스핀들)"])} |
| M (머신상태) | {", ".join(axis_groups["M (머신상태)"])} |

### 센서 의미 (축 공통)

| 접미사 | 의미 |
|--------|------|
| ActualPosition | 실제 위치 |
| ActualVelocity | 실제 속도 |
| ActualAcceleration | 실제 가속도 |
| SetPosition | 목표 위치 |
| SetVelocity | 목표 속도 |
| SetAcceleration | 목표 가속도 |
| CurrentFeedback | 전류 피드백 |
| DCBusVoltage | DC 버스 전압 |
| OutputCurrent | 출력 전류 |
| OutputVoltage | 출력 전압 |
| OutputPower | 출력 파워 (X/Y/Z 없음) |

### Machining_Process (공정 단계)

| 값 | 의미 |
|----|------|
| Starting | 시작 |
| Prep | 준비 |
| Layer 1 Up / Down | 1층 상승/하강 가공 |
| Layer 2 Up / Down | 2층 상승/하강 가공 |
| Layer 3 Up / Down | 3층 상승/하강 가공 |
| Repositioning | 재위치 조정 |
| end | 종료 |

---

## 모델링 전략 메모

- experiment 파일을 실험 단위로 집계(통계량 추출) → train.csv와 No로 조인
- 시계열 그대로 쓸 경우 실험마다 길이가 달라 패딩/트리밍 필요
- Machining_Process 별로 나눠 집계하는 것도 유효한 전략
"""

readme_path = PROJECT_DIR / "README_data.md"
readme_path.write_text(readme, encoding="utf-8")
print(f"\nREADME_data.md 생성 완료: {readme_path}")
