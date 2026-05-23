#!/usr/bin/env bash
set -euo pipefail

TEAM_ID="${1:-team1113}"
VERSION_NUM="${2:-1}"

if [ "$VERSION_NUM" = "final" ]; then
  TAG="${TEAM_ID}:final"
  OUT="${TEAM_ID}_final.tar.gz"
else
  TAG="${TEAM_ID}:v${VERSION_NUM}"
  OUT="${TEAM_ID}_v${VERSION_NUM}.tar.gz"
fi

cd "$(dirname "$0")/.."

echo "[submission] building image $TAG (linux/amd64)"
docker buildx build --platform linux/amd64 -t "$TAG" --load .

echo "[submission] saving to $OUT"
docker save "$TAG" | gzip -1 > "$OUT"

SIZE=$(du -h "$OUT" | cut -f1)
echo "[submission] done: $OUT  ($SIZE)"
echo "[submission] image:    $TAG"
echo "[submission] tarball:  $(pwd)/$OUT"
