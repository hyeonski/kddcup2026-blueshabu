# 다음 개선 계획

> 출발점: master HEAD `9205902`, 공개 50 태스크 평균 **0.5158** (`qwen/qwen3.5-35b-a3b`, 2026-04-30 측정).
> 목표: 1회차당 1~2개의 측정 가능한 개선을 출하. 각 작업은 현재 실행에서 관측된 특정 실패 버킷에 가설로 연결한다.

> ⚠️ **측정 분산 경고**: 동일 코드/데이터에서 단발 실행 점수가 0.42 ~ 0.52 범위로 흔들림 ([variance_2026-05-03.md](variance_2026-05-03.md)).
> OpenRouter provider routing이 주 원인으로 추정. `MODEL_PROVIDER_ONLY=alibaba` + `MODEL_SEED=42`로 pinning하면 Run 1 (0.5158) ↔ Run 3 (0.5081) 격차 0.008로 좁아짐.
> 아래 P0~P4 항목의 "타겟 태스크 ID"는 Run 1 스냅샷이며 매 실행마다 흔들릴 수 있음.
> **버킷 구조**(예: bucket A의 패턴)는 안정적이므로 plan의 방향성은 유효하되, 점수 효과 측정은 단발이 아닌 trace 레벨 검증을 우선시할 것.

## 1. 현재 위치

HEAD에서의 태스크별 버킷 ([baseline_9205902.md](baseline_9205902.md)):

| 버킷 | 태스크 수 | 점수 영향 |
| --- | ---: | ---: |
| 만점 (1.00) | 25 | 이미 최대 |
| 부분점 (0 < s < 1) | 4 | 여유 작음 |
| 0점 (예측 제출됨) | 14 | 모두 살리면 +0.28 |
| Prediction missing | 6 | 모두 살리면 +0.12 |
| Multi-row 부분 recall | 3 (부분점의 부분집합) | 작음 |

인프라 (env 모델 액세스, 채점기, 체크포인트, forced-answer)가 갖춰져 있어 어떤 변경이든 `dabench run-benchmark` + `dabench score`로 end-to-end 평가 가능.

## 2. 알려진 실패 모드

현재 실행의 trace 분석 + 점수표에서 도출:

| 모드 | 해당 태스크 | 진단 |
| --- | --- | --- |
| **A. Extra 컬럼 과잉 예측** | task_38 (5컬럼 vs gold 1), task_259 (4 vs 1), task_180 / 25 / 86 / 75 / 80 / 344 / 418 / 19 | 모델이 질문이 가리키는 단일 gold 컬럼 대신 **레코드 전체를 여러 컬럼**으로 출력. 순수 프롬프트/포맷 실패 — 값이 맞아도 점수 0. |
| **B. Single-column 잘못된 값** | task_145, 169, 199, 200, 163 | 1컬럼은 맞췄지만 내용이 틀림. 포맷이 아닌 추론 실패. |
| **C. Multi-row 부분 recall** | task_11 (1/3), task_27 (1/3), task_355 (1/3) | gold 3행인데 모델이 한 행만 찾고 나머지 누락. 너무 일찍 멈췄거나 필터를 너무 엄격하게 적용한 듯. |
| **D. Prediction missing** | task_173, 196, 250, 352, 379, 396, 89 | forced-answer가 시도됐지만 안착 못 함. 모델이 `answer`가 아닌 응답을 반환했거나, forced 단계에서 파싱 실패했거나, coerce할 final state 자체가 없었을 수 있음. |
| **E. `execute_python` 반복 로딩** | (커밋 `51439a0`에서 식별됨) | `execute_python` 호출마다 새 인터프리터 → 매 step CSV 재로드. 환경 셋업에 step 예산 소진. |
| **F. 대용량 컨텍스트 HTTP 타임아웃** | (커밋 `51439a0`에서 식별됨) | Hard / extreme 태스크 (>10K, >128K 토큰)가 60s HTTP 타임아웃에 걸림. 현재 chunking 전략 없음. |

## 3. 우선순위 작업 항목

각 항목은 측정 가능한 점수 델타와 함께 단일 PR로 출하 가능하도록 사이즈 조절. 우선순위는 **(예상 회수 점수) × (예상 공수⁻¹)** 기준.

### P0 — 프롬프트 레벨 컬럼 디시플린

**버킷 A + B 타겟 (현재 0점 ≥9 태스크).**

[agents/prompt.py](../src/data_agent_baseline/agents/prompt.py)의 시스템 프롬프트가 gold 컬럼 형태를 명시적으로 강제하지 않음. trace 분석 결과 모델이 기본적으로 "넓은" 출력을 선호.

구체 변경:

1. 시스템 프롬프트에 출력 형태 지시 추가:
   - "질문이 요구하는 컬럼만 정확히 출력. 추가 컬럼 1개당 점수가 λ × (extras / predicted)만큼 깎임."
   - "질문이 '엔티티 Y에 대한 X가 무엇인가'를 묻는다면 답은 **단일 컬럼**의 값이지 엔티티 레코드 전체가 아님."
2. 두 가지 실패 형태를 다루는 짧은 few-shot 2개 추가:
   - 단일 값 답 (1컬럼 × 1행).
   - 다중 행 열거 (1컬럼 × N행).
3. `answer` 호출 직전 self-check 강제:
   - "`answer` 호출 전, 다시 진술: 질문은 K컬럼 × N행 요구. 내 표는 K' × N'. K' ≠ K면 수정."

**측정:** 벤치마크 재실행, 0점(예측 제출됨) 카운트와 버킷 A를 구체적으로 비교. 목표: 버킷 A를 ≥50% 감축.

**리스크:** 과잉 교정 (multi-column gold에서 유효 컬럼을 버림) 가능. 예시를 균형 있게 유지해 완화.

**예상 공수:** 재실행 포함 1~2시간.

### P1 — `execute_python` 영속 상태

**버킷 E 타겟 + 간접적으로 C, F (Python 사용하는 모든 태스크의 step 예산 절약).**

지금은 [tools/python_exec.py](../src/data_agent_baseline/tools/python_exec.py)가 호출마다 새 subprocess를 시작. 모델이 매 step `import pandas as pd; df = pd.read_csv(...)`를 반복해야 함.

구체 변경:

1. 태스크별 `subprocess.Popen`으로 Python REPL (`-i -u`)을 띄우고, 단일 태스크 실행 동안 `execute_python` 호출 사이에 상태 유지.
2. 각 호출은 코드 + 센티넬을 stdin에 쓰고, 센티넬까지 stdout/stderr를 읽음.
3. 태스크 종료 시 리셋. 호출별 hard timeout은 그대로 강제.

이렇게 되면 모델이 한 step에 `df.head()`, 다음 step에 `df.groupby(...)`를 재로드 없이 할 수 있음.

**측정:** non-trivial 태스크당 평균 step 수가 약 30% 감소해야 함. 점수 영향: 간접적 — 실제 추론 step에 예산 확보. Hard 태스크 (~10K–128K)에서 소폭 직접 상승 기대.

**리스크:** 장기 실행 코드에서 REPL hang → 호출별 timeout 설정 후 kill + 재시작으로 묶음. 재시작 시 상태 손실은 어차피 그 step이 실패한 거라 허용.

**예상 공수:** 4~6시간 (REPL plumbing + crash/timeout/restart 테스트).

### P2 — 제출 전 self-verification 패스

**버킷 B 타겟 (답 형식은 맞지만 값이 틀린 5 태스크).**

`answer` 마무리 직전 검증 step 추가. 에이전트가 `answer` 호출 후:

1. 검증용 프롬프트 합성: 질문, 예측 표, 사용한 도구, 원시 값.
2. 모델에게 질의: "이 답이 질문에 부합하는가? 다르게 답할 거면 새 표 반환, 아니면 UNCHANGED."
3. 다른 표가 돌아오면 둘 다 로깅하고 새 것을 제출.

forced-answer와 차이: forced는 답이 **없을 때** 발동. 이건 답이 **있지만 틀릴 수도** 있을 때 발동.

**측정:** 버킷 B 태스크의 flip rate. 보수적 목표: 5개 중 2개를 만점으로 전환.

**리스크:** 검증자가 오답 생성한 동일 모델이라 그냥 도장 찍을 가능성. 신호를 높이기 위해 검증자에게 다른 framing ("엄격한 채점자라고 가정")을 주고, 변경만 수용하고 원본 거부는 절대 안 받도록.

**예상 공수:** 2~3시간.

### P3 — 진단: forced-answer 실패 분석

**버킷 D 타겟 (여전히 prediction missing인 6 태스크).**

forced-answer가 살려야 할 태스크들인데 못 살림. 추가 로직 전에 무슨 일이 일어나는지 계측 먼저. 6개 trace 각각 확인:

- forced step이 발동했나?
- 모델이 텍스트는 만들었지만 `answer` 호출 안 했나?
- forced 출력에서 파싱 복구가 실패했나?

산출물은 1쪽짜리 노트. 가능한 fix 카테고리:

- **Forced-answer 파싱 실패** → 원시 텍스트 → `answer(columns=[col], rows=[[value]])` 템플릿을 쓰는 더 엄격한 forced 프롬프트 추가.
- **Forced step 크래시** → [agents/react.py](../src/data_agent_baseline/agents/react.py) 견고성 fix.
- **태스크 자체가 실행 불가** → no-op, 0점 수용.

**측정:** 버킷 D ≤ 3 태스크 (6개 중 3개 구조).

**예상 공수:** 진단 2시간 + 수정 1~4시간.

### P4 — Hard / extreme 태스크용 long-context 처리

**버킷 F + extreme 2 태스크 타겟 (현재 거의 0점).**

규칙상 평가 시점에 보조 임베딩/검색 모델이 하드웨어 제한 내에서 허용됨. 대회 데이터엔 128K+ 토큰 문서가 있어 단순 `read_doc`으로는 부족.

구체 변경:

1. `read_doc` 동반 도구 추가: `search_doc(path, query, k=5)` — 첫 호출에 파일 chunk에 대한 in-memory FAISS / numpy 임베딩 인덱스 빌드, 태스크별 캐시.
2. 작은 CPU-friendly 임베딩 모델 (예: `BAAI/bge-small-en-v1.5`, ~33MB). RAM 안에 들어가고 CPU-only로 도는지 확인.
3. >10K 토큰 파일에 대해 `search_doc`을 사용하라고 프롬프트 갱신.

**측정:** 현재 0점인 extreme 2개 → 그 중 하나를 ≥0.5로 전환하는 게 현실적 목표. 추가로 hard에서 marginal 개선.

**리스크:** 의존성/메모리 무게, 태스크당 첫 호출 지연. 컨테이너 크기 예산 (10 GB 이미지 상한) 확인 필요.

**예상 공수:** 1~2일 (모델 다운로드, 인덱싱 파이프라인, 프롬프트 통합, 테스트).

## 4. 의도적 out-of-scope

- **Multi-agent / planner-executor 아키텍처** — 25-step 예산 대비 토큰 비용; 현재 실패 모드를 정당화하지 못하는 복잡도.
- **모델 fine-tuning** — 평가 시점 모델은 Qwen3.5-35B-A3B로 고정; 우리가 통제할 수 있는 건 prompt + tool + flow뿐.
- **평가용 LLM 제공자 교체** — 대회 규칙 위반; 로컬 dev에서만 교체 가능.
- **질문에서 추론하는 답 형식 휴리스틱** — 깨지기 쉬움; 프롬프트 엔지니어링이 더 일반적.

## 5. 프로세스

각 항목별:

1. `master`에서 분기. 항목 1개당 PR 1개.
2. 머지 전 50 태스크 벤치마크 실행. 결과를 `benchmarks/<branch>_<commit>.md` + `artifacts/runs/<branch>_<commit>/`에 저장.
3. 커밋 메시지에 포함: 가설, 메커니즘, 측정 점수 before / after, 어느 실패 버킷이 움직였는지.
4. 변경이 평균 점수를 올리지도 않고 후속 진단을 enable하지도 않는다면, 머지 전 revert.

## 6. 권장 순서

P0 → P3 (저렴, downside-free) → P1 → P2 → P4.

P0 먼저: 가장 큰 타겟 버킷에 대해 가장 저렴한 실험. P3 두 번째: 주로 진단이라 추가로 저렴한 win을 unlock할 수 있음. P1과 P2는 더 큰 구조적 변경; P4는 유일하게 런타임 의존성을 추가하므로 가장 신중한 통합 필요.
