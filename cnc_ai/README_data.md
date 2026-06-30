# CNC 데이터 구조 문서 (README_data.md)

자동 생성 by explore_data.py

---

## 파일 구성

| 파일 | 설명 | Shape |
|------|------|-------|
| train.csv | 실험별 메타정보 및 레이블 | 25행 × 8열 |
| experiment_01~25.csv | 실험별 센서 시계열 | 462~2332행 × 48열 |

---

## train.csv 컬럼 설명

| 컬럼 | 타입 | 설명 | 값 |
|------|------|------|-----|
| No | int | 실험 번호 | 1~25 |
| material | str | 가공 재료 | aluminum (전체 동일) |
| feedrate | int | 이송 속도 (mm/min) | [3, 6, 12, 15, 20] |
| clamp_pressure | float | 클램프 압력 (bar) | [2.5, 3.0, 4.0] |
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
| 양품(0) | 13 | 52.0% |
| 불량(1) | 12 | 48.0% |

### Task B: 공구마모 분류
- tool_condition 컬럼 직접 사용

| 클래스 | 수 | 비율 |
|--------|-----|------|
| unworn(0) | 11 | 44.0% |
| worn(1) | 14 | 56.0% |

---

## experiment_XX.csv 센서 컬럼 구조

총 32,048행 (25개 실험 합산), 실험별 462~2332행

### 축별 센서 그룹 (X/Y/Z/S 각 11개)

| 그룹 | 컬럼 |
|------|------|
| X축 (테이블 X방향) | X_ActualPosition, X_ActualVelocity, X_ActualAcceleration, X_SetPosition, X_SetVelocity, X_SetAcceleration, X_CurrentFeedback, X_DCBusVoltage, X_OutputCurrent, X_OutputVoltage, X_OutputPower |
| Y축 (테이블 Y방향) | Y_ActualPosition, Y_ActualVelocity, Y_ActualAcceleration, Y_SetPosition, Y_SetVelocity, Y_SetAcceleration, Y_CurrentFeedback, Y_DCBusVoltage, Y_OutputCurrent, Y_OutputVoltage, Y_OutputPower |
| Z축 (수직방향) | Z_ActualPosition, Z_ActualVelocity, Z_ActualAcceleration, Z_SetPosition, Z_SetVelocity, Z_SetAcceleration, Z_CurrentFeedback, Z_DCBusVoltage, Z_OutputCurrent, Z_OutputVoltage |
| S축 (스핀들) | S_ActualPosition, S_ActualVelocity, S_ActualAcceleration, S_SetPosition, S_SetVelocity, S_SetAcceleration, S_CurrentFeedback, S_DCBusVoltage, S_OutputCurrent, S_OutputVoltage, S_OutputPower, S_SystemInertia |
| M (머신상태) | M_CURRENT_PROGRAM_NUMBER, M_sequence_number, M_CURRENT_FEEDRATE |

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
