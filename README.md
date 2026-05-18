<div align="center">

# KDD Cup 2026 — team1113 Submission Repo

[![Official Website](https://img.shields.io/badge/Official%20Website-Visit%20dataagent.top-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=0f172a)](https://dataagent.top)
[![Demo Dataset](https://img.shields.io/badge/Demo%20Dataset-Download%20Phase%201-f59e0b?style=for-the-badge&logo=googledrive&logoColor=white&labelColor=0f172a)](https://drive.google.com/file/d/1c6u5WlFw4KV7CBRyXh5BvFYbKqxhBSbL/view)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white&labelColor=0f172a)](https://discord.com/invite/7eFwJQN3Fx)

</div>

> 팀 master 브랜치 기반의 KDD Cup 2026 DataAgent-Bench 제출 레포. HKUST DIAL 의
> [공식 starter kit](https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit) 을 fork 해 ReAct 에이전트와 ACON 컨텍스트 최적화, 도커 제출 패키징을 추가했다.
> 로컬에서는 `data/public/input/` 50 태스크에 ReAct 베이스라인을 돌려 점수를 측정하고,
> 제출은 [Submission (Docker)](#submission-docker) 의 amd64 컨테이너로 한다.

## Overview

| Item | Value |
| --- | --- |
| Local dataset | `data/public/input/` (50 demo tasks) |
| Public demo ground truth | `data/public/output/task_<id>/gold.csv` |
| Hidden eval (Phase 1) | A-board ≈ 60 tasks (2h limit) + B-board ≈ 320 tasks (12h total container budget) |
| Local entry | `uv run dabench <command> --config configs/react_baseline.example.yaml` |
| Submission entry | `bash scripts/make_submission.sh team1113 1` → `team1113_v1.tar.gz` |
| Local run output | `artifacts/runs/<run_id>/` |
| Docker run output | `/output/task_<id>/prediction.csv` (entrypoint 평탄화) + `/logs/` (사후 분석용) |

## Project Structure

```text
.
├── src/data_agent_baseline/   # ReAct 에이전트, 툴, 채점기, CLI (uv editable install)
│   ├── agents/                # react.py (루프), model.py (LLM), prompt.py
│   ├── benchmark/dataset.py   # 50 태스크 로더
│   ├── run/runner.py          # 단일/벤치마크 실행, subprocess 타임아웃, 체크포인트
│   ├── scoring/               # 정답 정규화 + Recall − λ·extras 채점기
│   ├── tools/                 # list_context, read_csv, execute_python, …
│   ├── cli.py                 # `dabench` 진입점 (typer)
│   └── config.py              # env > yaml > default 우선순위 해석
├── acon-main/                 # ACON HistoryOptimizer (path-source dep, productive-agents)
├── configs/
│   ├── react_baseline.example.yaml   # 로컬 개발 / 벤치마크용
│   └── submission.yaml               # 도커 컨테이너 내부에서 쓰이는 config (이미지 안에 굽힘)
├── docker/entrypoint.sh       # 컨테이너 시작 → run-benchmark → /output 평탄화 + /logs 보존
├── Dockerfile                 # python:3.11-slim 기반, pip editable install
├── scripts/
│   ├── make_submission.sh     # buildx amd64 + docker save | gzip → <team>_v<N>.tar.gz
│   └── test_run.sh            # 룰 §3.4 의 docker load + docker run 시뮬레이션
└── data/public/               # 공개 demo 입력/정답 (gitignored, 별도 다운로드)
```

## Quick Start

1. Install `uv` by following the official guide:
   - https://docs.astral.sh/uv/getting-started/installation/
2. On macOS and Linux, the standalone installer is:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Install project dependencies:

   ```bash
   uv sync
   ```

   `uv sync`는 `pyproject.toml`의 `[tool.uv.sources]`에 등록된 path 의존성을 따라
   `acon-main/`의 `productive-agents` 패키지를 editable 모드로 함께 설치한다.
   ACON 컨텍스트 최적화(`agent.enable_context_optimization`)를 켜면 이 패키지가 사용된다.

4. Set the model env vars. The grader injects these at evaluation time;
   set them locally for development too. See [Model access](#model-access)
   for provider-specific examples.

   ```bash
   # OpenRouter — same Qwen3.5-35B-A3B checkpoint the grader uses
   export MODEL_API_URL=https://openrouter.ai/api/v1
   export MODEL_API_KEY=sk-or-v1-...
   export MODEL_NAME=qwen/qwen3.5-35b-a3b

   # Recommended: pin OpenRouter routing + seed so cross-provider routing
   # doesn't add ±0.05 of variance to local benchmark scores
   export MODEL_PROVIDER_ONLY=alibaba
   export MODEL_SEED=42
   ```

   Tip: keep these in a project-local `.env` file (already git-ignored) and
   load with `set -a && . ./.env && set +a` before running. See
   [`.env.example`](.env.example) for ready-to-copy provider blocks.

   ⚠️ **두 가지 함정**:
   - **`MODEL_NAME` 정확히** — `qwen/qwen3.5-35b-a3b` (점 포함, 35b) 가 평가 환경 기준.
     `qwen/qwen3-35b-a3b` (점 없음, Qwen3 시리즈) 나 `qwen/qwen3-30b-a3b` (다른 사이즈) 는 다른 모델이라
     베이스라인 비교가 망가집니다. 우리 1차 도커 검증이 여기서 0.516 → 0.23 으로 폭락했음.
   - **`.env` 값에 따옴표 X** — `MODEL_API_URL="https://..."` 처럼 따옴표를 넣으면
     `docker --env-file` 가 따옴표째 변수에 박습니다. shell `export` 는 따옴표를 벗겨주지만
     `--env-file` 은 안 그렇습니다. [Submission (Docker)](#submission-docker) 흐름에서
     `awk` 로 따옴표 제거한 사본을 따로 만듭니다.

5. Confirm the dataset root and model env are wired up. The "Model Access"
   table should show `source=env` for the rows you exported:

   ```bash
   uv run dabench status --config configs/react_baseline.example.yaml
   ```

6. Smoke-test on a single task:

   ```bash
   uv run dabench run-task task_22 --config configs/react_baseline.example.yaml
   ```

7. Run the full public benchmark (50 tasks, ~17 min on Qwen3.5-35B-A3B
   via OpenRouter Alibaba):

   ```bash
   uv run dabench run-benchmark --config configs/react_baseline.example.yaml
   ```

8. Score the run against `data/public/output/*/gold.csv`. `$RUN_ID` 는 step 7
   의 결과 디렉토리 이름 — `react_baseline.example.yaml` 의 `run.run_id` 와
   동일 (default `qwen3.5-35b-a3b-ACON-Framework-baseline`):

   ```bash
   RUN_ID=qwen3.5-35b-a3b-ACON-Framework-baseline
   uv run dabench score $RUN_ID --config configs/react_baseline.example.yaml
   ```

9. (Optional) Inspect failures by hand. For any task that scored < 1.00,
   compare the model's prediction to the gold answer side-by-side:

   ```bash
   diff <(cat artifacts/runs/$RUN_ID/task_22/prediction.csv) \
        <(cat data/public/output/task_22/gold.csv)
   ```

   And read the full ReAct transcript to see what the model did:

   ```bash
   jq '.steps[] | {step: .step_index, action: .action, thought: .thought}' \
     artifacts/runs/$RUN_ID/task_22/trace.json
   ```

   See [Manual analysis workflow](#manual-analysis-workflow) below for more.

## Dataset

The public demo dataset lives under `data/public/input/`. Each task directory follows this structure:

```text
data/public/input/task_<id>/
├── task.json
└── context/
```

The corresponding public demo answers live separately under `data/public/output/task_<id>/gold.csv`.
Hidden test sets only include `input/`, so there is no `output/` directory there.

`task.json` contains:

- `task_id`
- `difficulty`
- `question`

The `context/` directory may contain one or more of:

- CSV files
- JSON files
- SQLite / DB files
- Text documents

## Model access

Resolution priority is **env > YAML > default**. The official evaluation
container injects `MODEL_API_URL`, `MODEL_API_KEY`, and `MODEL_NAME`, which
override anything in the YAML — so the same image runs locally and in the
grader without code changes. Hardcoding endpoints/keys is also explicitly
forbidden by the competition rules.

For local development, prefer env vars over editing the YAML.

**DeepSeek V3** (recommended commercial proxy for Qwen3.5-35B-A3B; ~$0.5
per 50-task benchmark):

```bash
export MODEL_API_URL=https://api.deepseek.com/v1
export MODEL_API_KEY=sk-...                # platform.deepseek.com
export MODEL_NAME=deepseek-chat            # V3; use deepseek-reasoner for R1
```

**OpenAI**:

```bash
export MODEL_API_URL=https://api.openai.com/v1
export MODEL_API_KEY=sk-...
export MODEL_NAME=gpt-4.1-mini             # or gpt-4o-mini
```

**OpenRouter** (recommended — hosts `qwen/qwen3.5-35b-a3b`, the same
checkpoint the official grader injects):

```bash
export MODEL_API_URL=https://openrouter.ai/api/v1
export MODEL_API_KEY=sk-or-v1-...
export MODEL_NAME=qwen/qwen3.5-35b-a3b

# Optional but strongly recommended for benchmarking. OpenRouter routes
# requests across multiple backends serving the same model id (different
# quantization, different inference engine), which adds non-deterministic
# variance to ReAct trajectories. Pin to a single provider + seed:
export MODEL_PROVIDER_ONLY=alibaba   # CSV; the model maker's own deployment
export MODEL_SEED=42                 # any int; constant across runs
```

**Local Qwen3.5-35B-A3B** (e.g. via vLLM in OpenAI-compatible mode):

```bash
export MODEL_API_URL=http://localhost:8000/v1
export MODEL_API_KEY=local
export MODEL_NAME=Qwen3.5-35B-A3B
```

`uv run dabench status --config <path>` prints a "Model Access" table that
shows whether each value came from `env`, `yaml`, or is `unset`. The API
key is never printed — only its length.

## Configuration

이 레포에는 두 개의 config 가 있고, **agent 설정은 동일**, 경로/run_id 만 다릅니다:

| 파일 | 용도 | 차이점 |
| --- | --- | --- |
| [`configs/react_baseline.example.yaml`](configs/react_baseline.example.yaml) | 로컬 개발/벤치마크 | `data/public/input` 읽고 `artifacts/runs/` 에 씀 |
| [`configs/submission.yaml`](configs/submission.yaml) | 도커 제출 (이미지 안에 굽힘) | `/input` 읽고 `/output/task_<id>/prediction.csv` 로 평탄화 — 자세한 흐름은 [Submission (Docker)](#submission-docker) |

새 파라미터를 튜닝할 때는 **둘 다 같이** 수정해야 로컬 점수와 도커 점수가 일치합니다.

`react_baseline.example.yaml` (현재 베이스라인 0.516 점수가 측정된 튜닝값):

```yaml
dataset:
  root_path: data/public/input

agent:
  # Model access priority: env > YAML > default.
  # During official evaluation, MODEL_API_URL / MODEL_API_KEY / MODEL_NAME
  # are injected by the grader and override these fields automatically.
  # Locally, prefer setting env vars instead of editing this file.
  model: MODEL NAME
  api_base: API BASE URL
  api_key: API KEY
  max_steps: 50
  temperature: 0.0
  # ACON Context Optimization
  enable_context_optimization: true
  history_summarization_threshold: 4000
  # wall_budget_seconds: null   # auto-derived: task_timeout_seconds - safety_margin
  # safety_margin_seconds: 30

run:
  output_dir: artifacts/runs
  run_id: qwen3.5-35b-a3b-ACON-Framework-baseline
  max_workers: 8
  task_timeout_seconds: 800
```

Config fields:

| Field | Meaning |
| --- | --- |
| `dataset.root_path` | Root directory of the public demo `input/` dataset. Relative paths are resolved from the project root. |
| `agent.model` | Model name. Overridden by env `MODEL_NAME` when set. |
| `agent.api_base` | OpenAI-compatible API base URL. Overridden by env `MODEL_API_URL` when set. |
| `agent.api_key` | API key. Overridden by env `MODEL_API_KEY` when set. Never commit real keys. |
| `agent.max_steps` | Maximum ReAct steps per task. |
| `agent.temperature` | Sampling temperature. 베이스라인은 `0.0` 으로 측정 — 1.0 으로 올리면 출력 분산이 커져 파싱 실패율이 오릅니다. |
| `agent.enable_context_optimization` | ACON HistoryOptimizer 사용 여부. 켜면 [`acon-main`](acon-main/) 의 `productive-agents` 패키지가 ReAct 히스토리를 동적으로 압축. |
| `agent.history_summarization_threshold` | ACON 이 히스토리 압축을 트리거하는 토큰 임계값. tiktoken 캐시가 없으면 `len(text)//4` 문자 근사로 폴백. |
| `agent.wall_budget_seconds` | Optional. Hard wall-clock budget the agent self-enforces, after which the loop exits gracefully and triggers a forced final-answer attempt. If null, derived as `run.task_timeout_seconds - agent.safety_margin_seconds`. |
| `agent.safety_margin_seconds` | Time reserved between the agent's wall budget and the runner's hard `task_timeout_seconds`. Default 30. |
| `agent.provider_only` | Optional. CSV or list of OpenRouter provider tags to allow (e.g. `alibaba`). Empty = no constraint. Overridden by env `MODEL_PROVIDER_ONLY`. Other OpenAI-compatible servers ignore this. |
| `agent.seed` | Optional. Int seed for sampling determinism on providers that honor it. Overridden by env `MODEL_SEED`. |
| `run.output_dir` | Output directory for run artifacts. |
| `run.run_id` | Optional run directory name. Defaults to a UTC timestamp if omitted. Must be a single directory name; existing run directories are rejected. |
| `run.max_workers` | Parallel worker count for `run-benchmark`. |
| `run.task_timeout_seconds` | Maximum wall-clock time per task. Set to `0` or a negative value to disable the task-level timeout. |

## CLI

```bash
uv run dabench <command> --config PATH [options]
```

| Command | Purpose | Example |
| --- | --- | --- |
| `status` | Show project paths, config path, dataset root, and public task counts. | `uv run dabench status --config configs/react_baseline.example.yaml` |
| `inspect-task` | Show task metadata and list accessible files under `context/`. | `uv run dabench inspect-task task_1 --config configs/react_baseline.example.yaml` |
| `run-task` | Run the baseline on one task and write outputs. | `uv run dabench run-task task_1 --config configs/react_baseline.example.yaml` |
| `run-benchmark` | Run the baseline across the public dataset. | `uv run dabench run-benchmark --config configs/react_baseline.example.yaml` |
| `score` | Score a completed run against `data/public/output/*/gold.csv`. | `uv run dabench score <run_id> --config configs/react_baseline.example.yaml` |

`run-benchmark` also supports `--limit N` to cap the number of tasks.

`score` reproduces the leaderboard formula
`max(0, Recall − λ × extras/predicted)` with column matching by
normalized, sorted value signatures. Override λ with `--lambda 0.5`. Values
are normalized (numeric → 2 decimal places, null variants → empty,
strings trimmed) before comparison, matching the rules at
https://dataagent.top/rules.

## Manual analysis workflow

After a `run-benchmark`, the score table tells you which tasks scored
< 1.00 but not why. The two artifacts to inspect for any task are
`prediction.csv` (what the model wrote) and `trace.json` (every step it
took). Compare them against `data/public/output/task_<id>/gold.csv` to
classify the failure.

### Quick triage from the score table

```bash
uv run dabench score $RUN_ID --config configs/react_baseline.example.yaml
```

In each row, `Matched/Gold` and `Pred` columns identify the failure mode:

| Pattern | Matched/Gold | Pred | Likely cause |
| --- | --- | --- | --- |
| Perfect | `K/K` | `K` | — |
| Extra columns | `K/K` | `> K` | Model emitted full record instead of single answer column |
| Wrong value | `0/1` | `1` | Reasoning failure — value content differs from gold |
| Partial recall | `< K/K` | `K` | Multi-row gold; model found subset |
| Prediction missing | `0/K` | `0`, note=`prediction missing` | `answer` never called; trace will show no terminal step |

### Diff a single task

```bash
TASK=task_22
diff <(cat artifacts/runs/$RUN_ID/$TASK/prediction.csv) \
     <(cat data/public/output/$TASK/gold.csv)
cat data/public/input/$TASK/task.json | jq .question
```

### Read the agent's trace

`trace.json` contains the full ReAct transcript. Useful queries:

```bash
# All thoughts + actions in order
jq '.steps[] | {step: .step_index, action: .action, thought: .thought}' \
  artifacts/runs/$RUN_ID/task_22/trace.json

# Just the final answer payload
jq '.answer' artifacts/runs/$RUN_ID/task_22/trace.json

# The last raw model response (useful when parse recovery fired)
jq '.steps[-1].raw_response' artifacts/runs/$RUN_ID/task_22/trace.json

# Failure reason if the runner caught an exception or timeout
jq '.failure_reason' artifacts/runs/$RUN_ID/task_22/trace.json
```

If the runner timed out, `trace.partial.json` is also written before
the kill — same shape as `trace.json`, recovered from the in-flight
checkpoint.

### Bucket failures across the run

```bash
# All "prediction missing" tasks
uv run dabench score $RUN_ID \
  --config configs/react_baseline.example.yaml | grep "missing"

# Tasks where Pred > Gold count (extra-column over-prediction)
for d in artifacts/runs/$RUN_ID/task_*; do
  task=$(basename $d)
  pred_cols=$(head -1 $d/prediction.csv 2>/dev/null | awk -F, '{print NF}')
  gold_cols=$(head -1 data/public/output/$task/gold.csv 2>/dev/null | awk -F, '{print NF}')
  [ -n "$pred_cols" ] && [ "$pred_cols" -gt "${gold_cols:-0}" ] \
    && echo "$task: pred=$pred_cols gold=$gold_cols"
done
```

## Tools

The baseline exposes these tools to the model:

| Tool | Purpose | Inputs |
| --- | --- | --- |
| `list_context` | List files and directories under `context/`. | `max_depth` |
| `read_csv` | Read a CSV preview. | `path`, `max_rows` |
| `read_json` | Read a JSON preview. | `path`, `max_chars` |
| `read_doc` | Read a text document preview. | `path`, `max_chars` |
| `inspect_sqlite_schema` | Inspect tables in a SQLite / DB file. | `path` |
| `execute_context_sql` | Execute read-only SQL against a SQLite / DB file in `context/`. | `path`, `sql`, `limit` |
| `execute_python` | Execute arbitrary Python code inside the task `context/` directory. | `code` |
| `answer` | Submit the final answer table and terminate the task. | `columns`, `rows` |

All file paths passed to tools must be relative to the task `context/` directory.

## Outputs

### 로컬 (`uv run dabench run-benchmark`)

```text
artifacts/runs/<run_id>/
├── summary.json                    # 태스크별 succeeded / failure_reason
├── task_<id>/
│   ├── trace.json                  # 전체 ReAct transcript (steps, observations, raw_response)
│   └── prediction.csv              # 정규화된 답안 (소수점 2자리, null=빈문자열)
└── ...
```

`task_timeout` 직전에 graceful exit 했다면 `task_<id>/checkpoint.json` 도 함께 남고, 다음 실행 시 부분 복구 가능.

### 도커 컨테이너 (제출용)

```text
/output/task_<id>/prediction.csv    # 평가 시스템이 채점하는 유일한 산출물
/logs/                               # 평가 시스템이 마운트해주는 사후 분석 영역
├── effective_config.yaml           # 컨테이너 안에서 실제 적용된 config (env override 반영)
├── run.log                         # dabench stdout/stderr
├── summary.json                    # 50 태스크 succeeded/failure 표
└── traces/task_<id>.trace.json     # 각 태스크 ReAct transcript
```

채점은 `prediction.csv` 만 본다 ([rules §4](https://dataagent.top/rules)). `/logs` 는 평가 후
호스트 디스크에 우리 팀 전용 영역으로 남아 다음 v2 튜닝의 1차 자료가 된다.

### 베이스라인 점수 레퍼런스

| Run | 평균 점수 | 측정 환경 | 문서 |
|---|---|---|---|
| 팀 master `9205902` | 0.5158 | 로컬 50 태스크, OpenRouter Alibaba | |
| 팀 v1 도커 (`bc650a4`) | ≈ 0.4954 | 도커 컨테이너 50 태스크 환산 | git log + `artifacts/eval/full-rebuild-1121/logs/summary.json` |
| upstream `c6992b0` | 0.2833 | 변경 없는 starter kit | |

분산 ±0.05 이내가 정상 범위. 새 변경이 이 범위 밖으로 떨어지면 회귀.

## Submission (Docker)

KDD Cup 2026 평가는 [`dataagent.top/rules`](https://dataagent.top/rules) §3.4 에
정의된 형식의 컨테이너로 진행된다. 평가 시스템은 다음 계약만 본다:

- **읽기**: `/input/task_<id>/...`
- **쓰기**: `/output/task_<id>/prediction.csv`
- **환경변수**: `MODEL_API_URL`, `MODEL_API_KEY`, `MODEL_NAME` (LLM 서비스 주소·키·모델명)

이 저장소는 위 계약을 만족하는 얇은 엔트리포인트와 amd64 Dockerfile을 함께
제공한다. 코드 자체는 이 세 환경변수를 이미 우선순위 1로 사용한다
([config.py](src/data_agent_baseline/config.py)).

### 1. 빌드

```bash
# 제출용 (linux/amd64) — 자동으로 docker save | gzip 까지 실행
bash scripts/make_submission.sh team1113 1
# → team1113:v1 이미지 + team1113_v1.tar.gz (≈ 300MB)

# (Apple Silicon 로컬 테스트용) arm64 네이티브 빌드
docker buildx build --platform linux/arm64 -t team1113:smoke --load .
```

`<team_id>` 는 운영진이 등록 직후 부여한 식별자(`teamNNNN`)와 정확히
일치해야 한다. `<N>` 은 제출 회차 (1부터 증가).

> **Apple Silicon 사용자 주의**: amd64 이미지(`team1113:v1`)를 Mac 에서 직접 실행하면
> qemu 에뮬레이션이 segfault 합니다 (`uv` / Python interpreter level). 로컬 검증은
> 별도로 빌드한 arm64 네이티브 이미지(`team1113:smoke`)로 하고, 제출용 amd64 는
> 평가 환경(실제 amd64 호스트)에서 동작 검증된다고 신뢰해야 합니다. 두 이미지 다
> 동일한 Dockerfile, 동일한 의존성 lock, 동일한 entrypoint 로 빌드되므로 행동은 같습니다.

### 2. 룰 형식 그대로 실행해 보기

[`scripts/test_run.sh`](scripts/test_run.sh) 가 §3.4 의 두 단계
(`docker load` → `docker run`) 를 그대로 수행한다. 로컬 사정상 다음
4가지가 룰 원문과 다르다:

| 항목 | 룰 | test_run.sh | 이유 |
|---|---|---|---|
| `--network=eval_net` | 있음 | 없음 | eval_net 은 평가 클러스터 내부망 — 로컬은 OpenRouter 외부망 필요 |
| `--cpus`, `--memory` | `16` / `64g` 고정 | `$CPUS`, `$MEMORY` 변수 (기본값 동일) | 호스트 자원이 부족할 때 override |
| `-e KEY=VAL` 3 개 | 명시 | `--env-file` | 키를 셸 히스토리에 노출 안 시키려고 |
| `--platform` | (eval 호스트가 amd64 라 불필요) | 명시 | Apple Silicon 빌드 매트릭스 |

```bash
# 환경변수 파일 준비 (값에 따옴표 X)
cat > artifacts/smoke/docker.env <<EOF
MODEL_API_URL=https://openrouter.ai/api/v1
MODEL_API_KEY=sk-or-v1-...
MODEL_NAME=qwen/qwen3.5-35b-a3b
EOF

# Apple Silicon
TEAM_ID=team1113 IMAGE_TAG=team1113:smoke \
PLATFORM=linux/arm64 CPUS=8 MEMORY=16g \
INPUT_DIR="$(pwd)/data/public/input" \
bash scripts/test_run.sh

# linux/amd64 호스트 (실제 평가환경과 동일)
TEAM_ID=team1113 IMAGE_TAG=team1113:v1 \
INPUT_DIR="$(pwd)/data/public/input" \
bash scripts/test_run.sh
```

성공하면 `artifacts/eval/<submission_id>/output/task_<id>/prediction.csv` 가
모든 입력 태스크에 대해 생성된다.

### 3. 파일 흐름

평가 시스템은 `/input` 에 모든 태스크를 한꺼번에 마운트한다.
[`docker/entrypoint.sh`](docker/entrypoint.sh) 는 다음을 수행한다:

1. `/app/configs/submission.yaml` 의 `dataset.root_path` 를 `/input` 으로,
   `run.output_dir` 을 `/tmp/runs` 로, `run.run_id` 를 `submission` 으로 덮어쓴
   임시 config 를 만든다.
2. `dabench run-benchmark` 를 임시 config 로 실행. stdout/stderr 는
   `tee /logs/run.log` 로 동시에 보존. 한 태스크가 실패해도 (`set +e` + `PIPESTATUS`
   체크로) 다음 단계가 계속 실행된다 — 부분 실패 격리.
3. `/tmp/runs/submission/task_<id>/prediction.csv` 를
   `/output/task_<id>/prediction.csv` 로 평탄화 복사 (한 단계 얕게).
4. `/logs` 가 마운트되어 있고 쓰기 가능하면 다음을 보존:
   - `effective_config.yaml` — env override 반영된 실제 적용 config
   - `summary.json` — 50 태스크 succeeded/failure_reason 표
   - `traces/task_<id>.trace.json` — 각 태스크 전체 ReAct transcript

채점은 `/output/task_<id>/prediction.csv` 만 본다 — `/logs` 에 무엇을 쓰든
점수에 영향 없다 ([rules §4](https://dataagent.top/rules)). `/logs` 는 평가 후
**우리 팀 전용** 영역으로 남아 다음 제출 튜닝의 디버깅 자료가 된다 (다른 팀과 공유 X).

### 4. 제출

1. **사전 확인** — 제출 전에 매번 체크:
   - `team_id` 가 운영진 부여 ID 와 정확히 일치 (registration confirmation 메일)
   - 오늘 우리 팀의 다른 제출이 없음 (일일 1회 제한)
   - 이전 제출의 평가가 완료됨 (queue 가 비어 있음)
2. **Google Drive 업로드** — `team1113_v1.tar.gz` → "링크가 있는 모든 사용자: 뷰어"
3. **운영진 메일** — 등록 시 기재된 팀 리더 이메일에서:
   - 제목: `[KDDCup2026 Data Agents] Submission - team1113 - v1`
   - 본문: 팀 ID, 버전, 공유 링크

**Phase 1 제출 한도** ([rules §3](https://dataagent.top/rules)):
- 일일 1회, 총 30회 (Phase 1 전체)
- A-board: 단일 평가 wall-clock **2 시간** (≈ 60 태스크)
- B-board: 컨테이너 전체 **12 시간** (≈ 320 태스크) — A-board 와 합산
- 380 태스크 worst-case (모두 800 s timeout) 시 `max_workers=8` 환경에서 ≈ 7.92 h → 12 h 안에 들어옴.
  A-board worst-case (60 × 800 / 8) 는 ≈ 1.67 h → 2 h 안.

### 5. 룰 적합성 체크리스트

- [x] 이미지 이름 `<team_id>:v<N>`, 아카이브 `<team_id>_v<N>.tar.gz`
      (콜론을 언더스코어로) — `make_submission.sh` 가 자동 처리
- [x] linux/amd64 (10 GB 이내, 현재 ≈ 300 MB)
- [x] `docker save … | gzip` 으로 생성된 tar.gz
- [x] `MODEL_API_URL/KEY/NAME` 을 코드에 하드코딩하지 않음 (env 우선)
- [x] `/input` 미수정, `/output/task_<id>/prediction.csv` 만 기록
- [x] 외부 LLM 호출은 평가 단계에서 차단됨 — Qwen3.5-35B-A3B 만 호출

## Contact

- Open issues: https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit/issues
- Official website: https://dataagent.top
- Discord: https://discord.com/invite/7eFwJQN3Fx
- WeChat official account: `数据智能与分析实验室 DIAL`

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://dataagent.top">
          <img
            src="https://api.qrserver.com/v1/create-qr-code/?size=144x144&data=https://dataagent.top&bgcolor=ffffff&color=111827&margin=8"
            alt="Official website QR code"
            width="144"
          />
        </a>
        <br />
        Official Website
      </td>
      <td align="center">
        <a href="https://discord.com/invite/7eFwJQN3Fx">
          <img
            src="https://api.qrserver.com/v1/create-qr-code/?size=144x144&data=https://discord.com/invite/7eFwJQN3Fx&bgcolor=ffffff&color=111827&margin=8"
            alt="Discord QR code"
            width="144"
          />
        </a>
        <br />
        Discord
      </td>
      <td align="center">
        <img
          src="https://dataagent.top/HKUSTGZ_DIAL.jpg"
          alt="WeChat official account QR code"
          width="144"
        />
        <br />
        WeChat Official Account
      </td>
    </tr>
  </table>
</div>

## Main Modules

| Module | Responsibility |
| --- | --- |
| `src/data_agent_baseline/cli.py` | `dabench` 진입점 (typer): `status`, `inspect-task`, `run-task`, `run-benchmark`, `score` |
| `src/data_agent_baseline/config.py` | `AppConfig` / `AgentConfig` / `RunConfig` — env > yaml > default 우선순위 해석 |
| `src/data_agent_baseline/benchmark/dataset.py` | `DABenchPublicDataset` — 50 태스크 로더 |
| `src/data_agent_baseline/benchmark/schema.py` | `PublicTask` 데이터 클래스 |
| `src/data_agent_baseline/agents/react.py` | ReAct 루프: thought/action JSON 프로토콜, wall budget, ACON 압축 |
| `src/data_agent_baseline/agents/runtime.py` | `AgentRuntimeState`, `StepRecord`, `AgentRunResult` |
| `src/data_agent_baseline/agents/model.py` | OpenAI 호환 LLM 어댑터 (provider_only, seed) + ACON LLM wrapper |
| `src/data_agent_baseline/agents/prompt.py` | System / task / observation 프롬프트 빌더 |
| `src/data_agent_baseline/run/runner.py` | 단일 태스크 / 벤치마크 실행: `multiprocessing.Process` 격리, 체크포인트 복구, 강제 final-answer fallback |
| `src/data_agent_baseline/scoring/normalize.py` | 정답 정규화 (소수점 2자리, null 변형 통일, 날짜 ISO 8601) |
| `src/data_agent_baseline/scoring/score.py` | Recall − λ × extras 채점기, 컬럼 시그니처 매칭 |
| `src/data_agent_baseline/tools/registry.py` | `ToolRegistry`, `ToolSpec`, terminal `answer` |
| `src/data_agent_baseline/tools/filesystem.py` | `list_context`, `read_csv`, `read_json`, `read_doc` |
| `src/data_agent_baseline/tools/python_exec.py` | `execute_python` (sandboxed `context/` cwd) |
| `src/data_agent_baseline/tools/sqlite.py` | `inspect_sqlite_schema`, `execute_context_sql` |
| `acon-main/src/productive_agents/ctxopt/history_optimizer.py` | ReAct 히스토리 토큰 임계 도달 시 LLM 으로 압축 |
| `docker/entrypoint.sh` | 컨테이너 진입점: 임시 config → run-benchmark → /output 평탄화 → /logs 보존 |
