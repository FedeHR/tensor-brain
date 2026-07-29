#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/common.sh"

cd "$TB_REPO_ROOT"
uv run --frozen --no-sync python -m experiments.pvsg.materialize \
  --dataset-root "$PVSG_DATASET_ROOT" \
  --extraction-manifest "$PVSG_MANIFEST_ROOT/videos.jsonl" \
  --feature-root "$DINO_FEATURE_ROOT" \
  --output-root "$PVSG_SECTION6_MANIFEST_ROOT"
