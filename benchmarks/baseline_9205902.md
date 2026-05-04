# 베이스라인 벤치마크 — 커밋 `9205902`

> 공개 50 태스크 데모 셋에서 수정 없이 돌린 ReAct 베이스라인의 스냅샷.
> 에이전트 변경의 효과를 측정할 때 이 결과를 기준점으로 삼는다.

## 실행 환경

| 항목 | 값 |
| --- | --- |
| 커밋 | `9205902` (Document env-driven model setup and full local workflow in README) |
| 모델 | `qwen/qwen3.5-35b-a3b` (OpenRouter, 공식 채점기와 동일 체크포인트) |
| 설정 | `configs/react_baseline.example.yaml` (max_steps=25, temperature=0.0, max_workers=8, task_timeout=600s) |
| 데이터셋 | `data/public/input/` — 50 태스크 (easy=15, medium=23, hard=10, extreme=2) |
| λ (extras 페널티) | 0.5 |
| 측정 일자 | 2026-04-30 |

## 핵심 지표

> ⚠️ **이 0.5158은 단발 측정**이며 다른 날 같은 코드로 다시 돌리면 0.42 ~ 0.52 범위에서 흔들림. 분산 분석은 [variance_2026-05-03.md](variance_2026-05-03.md). pinning 적용 시 Run 3 = 0.5081로 본 결과와 0.008 차이.

| 지표 | 값 |
| --- | --- |
| **평균 점수** | **0.5158** |
| 만점(1.00) 태스크 | 25 / 50 |
| 부분점 태스크 | 4 / 50 |
| 0점 (예측은 제출됨) | 15 / 50 |
| `prediction missing` | 6 / 50 |
| 러너 성공률 (예외 없이 종료) | 39 / 50 |
| 실행 시간 | 약 17분 (2.9 task/min) |

## 태스크별 점수

전체 표는 [scores.txt](../artifacts/runs/baseline_9205902_qwen35a3b/scores.txt) 참고.

### 점수 버킷

- **만점(1.00)** — task_22, 24, 26, 64, 67, 74, 194, 214, 218, 243, 249, 257, 261, 269, 283, 287, 292, 303, 305, 330, 349, 350, 408, 415, 420
- **부분점** — task_11 (0.00, 1/3 recall + extras), task_27 (0.08), task_259 (0.62), task_355 (0.08)
- **0점 (예측 제출됨)** — task_19, 25, 38, 75, 80, 86, 145, 163, 169, 180, 199, 200, 344, 418
- **prediction missing** — task_173, 196, 250, 352, 379, 396, 89

## 공략할 실패 패턴

1. **Prediction missing (6 태스크)** — 에이전트가 `answer`를 호출하지 않고 종료. 강제 final-answer fallback이 왜 못 살렸는지 trace 분석 필요.
2. **Extra 컬럼 과잉 예측** — task_38 (5컬럼 vs gold 1), task_259 (4 vs 1), task_180/25/86 (single-col gold에 extras=1.0).
3. **Single-column 단순 오답** — task_145, 169, 199, 200, 344, 418, 75, 80 — 답 형식은 맞췄으나 값이 틀림. self-check 또는 grounding으로 잡힐 가능성.
4. **Multi-row 부분 recall** — task_11, 27, 355 — gold 3행인데 모델이 1행만 찾음.

## 산출물

- 실행 디렉터리: `artifacts/runs/baseline_9205902_qwen35a3b/`
  - `summary.json` — 태스크별 러너 성공/실패
  - `task_<id>/prediction.csv` — 모델 출력
  - `task_<id>/trace.json` — ReAct 전체 transcript (실패 분석용)
  - `scores.txt` — 공식 공식 채점 표
  - `run.log` — 벤치마크 stdout/stderr
