#!/usr/bin/env bash
# Mirrors Section 3.4 of https://dataagent.top/rules.
# Step 1: docker load
# Step 2: docker run with eval-equivalent flags
#
# Local-only deviations (documented):
#   * --network=eval_net is omitted — that network exists only inside the
#     evaluation cluster and blocks egress; locally we need outbound access
#     to OpenRouter.
#   * On Apple Silicon, the linux/amd64 image qemu-segfaults; pass IMAGE_TAG
#     to point at an arm64-built image (blueshabu:smoke) for local smoke,
#     or run on a real linux/amd64 host to use blueshabu:v1.
#   * /logs is mounted but the current solution doesn't write there yet.
set -euo pipefail

cd "$(dirname "$0")/.."

TEAM_ID="${TEAM_ID:-team1113}"
VERSION="${VERSION:-1}"
IMAGE_TAG="${IMAGE_TAG:-${TEAM_ID}:v${VERSION}}"
TARBALL="${TARBALL:-${TEAM_ID}_v${VERSION}.tar.gz}"
SUBMISSION_ID="${SUBMISSION_ID:-local-$(date +%s)}"

INPUT_DIR="${INPUT_DIR:-$(pwd)/data/public/input}"
EVAL_ROOT="${EVAL_ROOT:-$(pwd)/artifacts/eval}"
OUTPUT_DIR="$EVAL_ROOT/$SUBMISSION_ID/output"
LOGS_DIR="$EVAL_ROOT/$SUBMISSION_ID/logs"

ENV_FILE="${ENV_FILE:-artifacts/smoke/docker.env}"
PLATFORM="${PLATFORM:-linux/amd64}"
CPUS="${CPUS:-16}"        # rules: 16. Override locally if host has fewer.
MEMORY="${MEMORY:-64g}"   # rules: 64g

if [ ! -f "$ENV_FILE" ]; then
  echo "[test_run] ENV_FILE $ENV_FILE missing." >&2
  echo "  Create one with MODEL_API_URL/MODEL_API_KEY/MODEL_NAME (no quotes)." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

# Step 1: docker load -i <team>_v<N>.tar.gz
if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  if [ -f "$TARBALL" ]; then
    echo "[test_run] docker load -i $TARBALL"
    docker load -i "$TARBALL"
  else
    echo "[test_run] $IMAGE_TAG not present and $TARBALL missing." >&2
    exit 3
  fi
else
  echo "[test_run] image $IMAGE_TAG already loaded; skipping docker load"
fi

# Step 2: docker run (Section 3.4)
echo "[test_run] docker run  image=$IMAGE_TAG  submission_id=$SUBMISSION_ID  platform=$PLATFORM"
set -x
docker run --rm \
  --platform "$PLATFORM" \
  --cpus="$CPUS" \
  --memory="$MEMORY" \
  --memory-swap="$MEMORY" \
  --env-file "$ENV_FILE" \
  -v "$INPUT_DIR:/input:ro" \
  -v "$OUTPUT_DIR:/output:rw" \
  -v "$LOGS_DIR:/logs:rw" \
  "$IMAGE_TAG"
set +x

echo "[test_run] outputs:"
find "$OUTPUT_DIR" -maxdepth 2 -type f | sort
