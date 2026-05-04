<div align="center">

# DataAgent-Bench Starter Kit

[![Official Website](https://img.shields.io/badge/Official%20Website-Visit%20dataagent.top-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=0f172a)](https://dataagent.top)
[![Demo Dataset](https://img.shields.io/badge/Demo%20Dataset-Download%20Phase%201-f59e0b?style=for-the-badge&logo=googledrive&logoColor=white&labelColor=0f172a)](https://drive.google.com/file/d/1c6u5WlFw4KV7CBRyXh5BvFYbKqxhBSbL/view)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white&labelColor=0f172a)](https://discord.com/invite/7eFwJQN3Fx)

</div>

> Official starter kit for the KDD Cup 2026 DataAgent-Bench challenge. The repository reads tasks from `data/public/input/` and writes predictions for downstream evaluation.

## Overview

| Item | Value |
| --- | --- |
| Dataset input | `data/public/input/` |
| Public demo ground truth | `data/public/output/task_<id>/gold.csv` |
| Hidden test data | `input/` only, no `output/` |
| Entry command | `uv run dabench <command> --config PATH` |
| Default run output | `artifacts/runs/` |

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
   # (see benchmarks/variance_2026-05-03.md)
   export MODEL_PROVIDER_ONLY=alibaba
   export MODEL_SEED=42
   ```

   Tip: keep these in a project-local `.env` file (already git-ignored) and
   load with `set -a && . ./.env && set +a` before running. See
   [`.env.example`](.env.example) for ready-to-copy provider blocks.

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

8. Score the run against `data/public/output/*/gold.csv`:

   ```bash
   uv run dabench score example_run_id --config configs/react_baseline.example.yaml
   ```

9. (Optional) Inspect failures by hand. For any task that scored < 1.00,
   compare the model's prediction to the gold answer side-by-side:

   ```bash
   diff <(cat artifacts/runs/example_run_id/task_22/prediction.csv) \
        <(cat data/public/output/task_22/gold.csv)
   ```

   And read the full ReAct transcript to see what the model did:

   ```bash
   jq '.steps[] | {step: .step_index, action: .action, thought: .thought}' \
     artifacts/runs/example_run_id/task_22/trace.json
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

An example config file lives at `configs/react_baseline.example.yaml`.

```yaml
dataset:
  root_path: data/public/input

agent:
  # Env vars MODEL_API_URL / MODEL_API_KEY / MODEL_NAME override these.
  model: YOUR_MODEL_NAME
  api_base: YOUR_API_BASE_URL
  api_key: YOUR_API_KEY
  max_steps: 25
  temperature: 0.0
  # wall_budget_seconds: null   # auto-derived from task_timeout_seconds - safety_margin
  # safety_margin_seconds: 30

run:
  output_dir: artifacts/runs
  run_id: example_run_id
  max_workers: 4
  task_timeout_seconds: 600
```

Config fields:

| Field | Meaning |
| --- | --- |
| `dataset.root_path` | Root directory of the public demo `input/` dataset. Relative paths are resolved from the project root. |
| `agent.model` | Model name. Overridden by env `MODEL_NAME` when set. |
| `agent.api_base` | OpenAI-compatible API base URL. Overridden by env `MODEL_API_URL` when set. |
| `agent.api_key` | API key. Overridden by env `MODEL_API_KEY` when set. Never commit real keys. |
| `agent.max_steps` | Maximum ReAct steps per task. |
| `agent.temperature` | Sampling temperature. |
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
| `inspect-task` | Show task metadata and list accessible files under `context/`. | `uv run dabench inspect-task task_1 --config configs/react_baseline.local.yaml` |
| `run-task` | Run the baseline on one task and write outputs. | `uv run dabench run-task task_1 --config configs/react_baseline.local.yaml` |
| `run-benchmark` | Run the baseline across the public dataset. | `uv run dabench run-benchmark --config configs/react_baseline.local.yaml` |
| `score` | Score a completed run against `data/public/output/*/gold.csv`. | `uv run dabench score <run_id> --config configs/react_baseline.local.yaml` |

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
uv run dabench score example_run_id --config configs/react_baseline.example.yaml
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
diff <(cat artifacts/runs/example_run_id/$TASK/prediction.csv) \
     <(cat data/public/output/$TASK/gold.csv)
cat data/public/input/$TASK/task.json | jq .question
```

### Read the agent's trace

`trace.json` contains the full ReAct transcript. Useful queries:

```bash
# All thoughts + actions in order
jq '.steps[] | {step: .step_index, action: .action, thought: .thought}' \
  artifacts/runs/example_run_id/task_22/trace.json

# Just the final answer payload
jq '.answer' artifacts/runs/example_run_id/task_22/trace.json

# The last raw model response (useful when parse recovery fired)
jq '.steps[-1].raw_response' artifacts/runs/example_run_id/task_22/trace.json

# Failure reason if the runner caught an exception or timeout
jq '.failure_reason' artifacts/runs/example_run_id/task_22/trace.json
```

If the runner timed out, `trace.partial.json` is also written before
the kill — same shape as `trace.json`, recovered from the in-flight
checkpoint.

### Bucket failures across the run

```bash
# All "prediction missing" tasks
uv run dabench score example_run_id \
  --config configs/react_baseline.example.yaml | grep "missing"

# Tasks where Pred > Gold count (extra-column over-prediction)
for d in artifacts/runs/example_run_id/task_*; do
  task=$(basename $d)
  pred_cols=$(head -1 $d/prediction.csv 2>/dev/null | awk -F, '{print NF}')
  gold_cols=$(head -1 data/public/output/$task/gold.csv 2>/dev/null | awk -F, '{print NF}')
  [ -n "$pred_cols" ] && [ "$pred_cols" -gt "${gold_cols:-0}" ] \
    && echo "$task: pred=$pred_cols gold=$gold_cols"
done
```

### Reference snapshots

- [`benchmarks/baseline_9205902.md`](benchmarks/baseline_9205902.md) — bucket-level
  analysis of the team master HEAD with this exact workflow.
- [`benchmarks/variance_2026-05-03.md`](benchmarks/variance_2026-05-03.md) —
  cross-run variance and how `MODEL_PROVIDER_ONLY=alibaba` reduces it.
- [`benchmarks/plan_next_improvements.md`](benchmarks/plan_next_improvements.md) —
  prioritized work items mapped to the failure buckets above.

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

Each successful task run may produce:

- `trace.json`
- `prediction.csv`

Per-task outputs are written to:

```text
artifacts/runs/<run_id>/<task_id>/
├── trace.json
└── prediction.csv
```

Benchmark runs also write:

```text
artifacts/runs/<run_id>/summary.json
```

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
| `src/data_agent_baseline/benchmark/dataset.py` | Public dataset loader |
| `src/data_agent_baseline/tools/filesystem.py` | `list_context`, `read_csv`, `read_json`, `read_doc` |
| `src/data_agent_baseline/tools/python_exec.py` | `execute_python` |
| `src/data_agent_baseline/tools/sqlite.py` | `inspect_sqlite_schema`, `execute_context_sql` |
| `src/data_agent_baseline/tools/registry.py` | Tool registration and terminal `answer` |
| `src/data_agent_baseline/agents/prompt.py` | System prompt, task prompt, observation prompt |
| `src/data_agent_baseline/agents/react.py` | ReAct runtime with JSON action protocol |
| `src/data_agent_baseline/run/runner.py` | Single-task and benchmark execution |
