# 주최측 베이스라인 벤치마크 — 커밋 `c6992b0`

> **주최측이 제공한 원본** ReAct 베이스라인 (팀 변경 없음)의 스냅샷.
> 우리 팀 커밋의 효과를 측정하는 기준점.

## 실행 환경

| 항목 | 값 |
| --- | --- |
| 커밋 | `c6992b0` (docs: update phase1 dataset links — 우리 팀 변경 직전 마지막 upstream 커밋) |
| 모델 | `qwen/qwen3.5-35b-a3b` (OpenRouter) |
| 설정 | `configs/react_baseline.example.yaml` (max_steps=**16**, temperature=0.0, max_workers=8, task_timeout=600s) |
| 데이터셋 | 동일한 `data/public/input/`을 worktree(`/tmp/kddcup-baseline-c6992b0/`)에 심볼릭 링크 |
| λ (extras 페널티) | 0.5 |
| 측정 일자 | 2026-04-30 |

`c6992b0`용 `git worktree`에서 실행해 현재 작업 트리는 건드리지 않음. 채점은 **현재 레포의** `dabench score`를 사용 (베이스라인 커밋 시점엔 로컬 채점기가 없음).

## 핵심 지표

| 지표 | 베이스라인 `c6992b0` | 팀 master `9205902` | Δ |
| --- | --- | --- | --- |
| **평균 점수** | **0.2833** | **0.5158** | **+0.2325** |
| 만점(1.00) 태스크 | 14 / 50 | 25 / 50 | +11 |
| `prediction missing` | 점수표상 약 27행¹ | 6 / 50 | −21 |
| 러너 성공률 | 23 / 50 | 39 / 50 | +16 |

¹ "missing" 카운트는 점수표의 줄바꿈된 multi-line 행을 포함한 수치. 러너 레벨 통계로는 **23 succeeded / 50 attempted = 27 실패** vs 팀의 11 실패.

## 우리 커밋이 무엇을 추가했나 (vs 베이스라인)

`c6992b0`와 `9205902` 사이의 4개 커밋:

1. `0cb0d8b` — env 기반 모델 액세스 (`MODEL_API_URL/KEY/NAME`), ReAct 파싱 복구, 로컬 채점기
2. `51439a0` — 타임아웃용 wall budget + 체크포인트 복구
3. `9e76fc3` — **에이전트가 답 제출 없이 종료할 때 강제 final-answer 시도**
4. `9205902` — README 문서화

가장 큰 효자는 `9e76fc3` (안 그러면 `prediction missing`이 됐을 태스크들을 살림)와 `51439a0` (graceful 타임아웃 처리). 러너 성공이 23 → 39 (+16)로 점프한 것이 이 두 fix와 거의 일치.

또한: 팀 설정에서 `max_steps`를 16 → 25로 올려 다단계 태스크에 여유를 줌.

## 산출물

- 실행 디렉터리: `artifacts/runs/baseline_c6992b0_qwen35a3b/`
  - `summary.json`, `scores.txt`, `run.log`
  - `task_<id>/prediction.csv` + `task_<id>/trace.json` (50 태스크)
- 소스 worktree (디스크에 보존): `/tmp/kddcup-baseline-c6992b0/`

## 태스크별 점수

전체 표는 [scores.txt](../artifacts/runs/baseline_c6992b0_qwen35a3b/scores.txt) 참고.
