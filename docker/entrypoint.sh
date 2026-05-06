#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-/app/configs/submission.yaml}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-/tmp/runs}"
RUN_ID="${RUN_ID:-submission}"
INPUT_DIR="${INPUT_DIR:-/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"

if [ ! -d "$INPUT_DIR" ]; then
  echo "[entrypoint] FATAL: input directory $INPUT_DIR not found" >&2
  exit 2
fi

if [ -z "${MODEL_API_URL:-}" ] || [ -z "${MODEL_NAME:-}" ]; then
  echo "[entrypoint] WARNING: MODEL_API_URL or MODEL_NAME is empty; the agent will fall back to YAML/defaults." >&2
fi

mkdir -p "$OUTPUT_DIR" "$RUN_OUTPUT_ROOT"

echo "[entrypoint] dataset=$INPUT_DIR  output=$OUTPUT_DIR  config=$CONFIG_PATH"
echo "[entrypoint] tasks discovered: $(ls "$INPUT_DIR" | wc -l)"

# run-benchmark reads dataset.root_path / run.output_dir from YAML;
# we override via env-aware wrapper that writes a temp config.
TMP_CFG="$(mktemp -t submission-XXXXXX.yaml)"
python - "$CONFIG_PATH" "$TMP_CFG" "$INPUT_DIR" "$RUN_OUTPUT_ROOT" "$RUN_ID" <<'PY'
import sys, yaml, pathlib
src, dst, input_dir, output_root, run_id = sys.argv[1:6]
cfg = yaml.safe_load(pathlib.Path(src).read_text()) or {}
cfg.setdefault("dataset", {})["root_path"] = input_dir
cfg.setdefault("run", {})
cfg["run"]["output_dir"] = output_root
cfg["run"]["run_id"] = run_id
pathlib.Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

echo "[entrypoint] effective config:"
cat "$TMP_CFG"

# /logs is mounted rw by the evaluation system per rules §3.4. We use it to
# preserve stdout/stderr and per-task ReAct traces so post-eval debugging is
# possible — scoring still only reads /output/task_<id>/prediction.csv.
LOGS_DIR="${LOGS_DIR:-/logs}"
if [ -d "$LOGS_DIR" ] && [ -w "$LOGS_DIR" ]; then
  cp "$TMP_CFG" "$LOGS_DIR/effective_config.yaml" || true
  RUN_LOG="$LOGS_DIR/runtime.log"
else
  RUN_LOG="/tmp/run.log"
fi

echo "[entrypoint] $ dabench run-benchmark --config $TMP_CFG  (tee → $RUN_LOG)"
set +e
dabench run-benchmark --config "$TMP_CFG" 2>&1 | tee "$RUN_LOG"
RC=${PIPESTATUS[0]}
set -e
echo "[entrypoint] dabench exit code: $RC"

RUN_DIR="$RUN_OUTPUT_ROOT/$RUN_ID"
if [ ! -d "$RUN_DIR" ]; then
  echo "[entrypoint] FATAL: expected run dir $RUN_DIR missing (rc=$RC)" >&2
  exit 3
fi

echo "[entrypoint] flattening predictions into $OUTPUT_DIR"
copied=0
for src_csv in "$RUN_DIR"/task_*/prediction.csv; do
  [ -e "$src_csv" ] || continue
  task_dir="$(basename "$(dirname "$src_csv")")"
  dst_dir="$OUTPUT_DIR/$task_dir"
  mkdir -p "$dst_dir"
  cp "$src_csv" "$dst_dir/prediction.csv"
  copied=$((copied + 1))
done
echo "[entrypoint] wrote $copied prediction.csv files"

# Preserve per-task traces and the run summary in /logs for post-eval debugging.
if [ -d "$LOGS_DIR" ] && [ -w "$LOGS_DIR" ]; then
  trace_count=0
  mkdir -p "$LOGS_DIR/traces"
  for trace in "$RUN_DIR"/task_*/trace.json; do
    [ -e "$trace" ] || continue
    task_dir="$(basename "$(dirname "$trace")")"
    cp "$trace" "$LOGS_DIR/traces/${task_dir}.trace.json" || true
    trace_count=$((trace_count + 1))
  done
  cp "$RUN_DIR/summary.json" "$LOGS_DIR/summary.json" 2>/dev/null || true
  echo "[entrypoint] copied $trace_count traces + summary into $LOGS_DIR"
fi
