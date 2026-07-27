#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/common.sh"

MANIFEST="$PVSG_MANIFEST_ROOT/videos.jsonl"
if [[ ! -s "$MANIFEST" ]]; then
  echo "Missing extraction manifest: $MANIFEST. Run cluster/pvsg/setup.sh first." >&2
  exit 2
fi

NUM_VIDEOS="$(wc -l < "$MANIFEST")"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
if [[ ! "$NUM_VIDEOS" =~ ^[1-9][0-9]*$ || ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "Manifest size and MAX_PARALLEL must be positive integers." >&2
  exit 2
fi
LAST_INDEX="$((NUM_VIDEOS - 1))"
ARRAY_RANGE="${ARRAY_RANGE:-0-${LAST_INDEX}%${MAX_PARALLEL}}"

mkdir -p "$SLURM_LOG_ROOT" "$DINO_FEATURE_ROOT"

sbatch \
  --array="$ARRAY_RANGE" \
  --output="$SLURM_LOG_ROOT/%x-%A_%a.out" \
  --error="$SLURM_LOG_ROOT/%x-%A_%a.err" \
  --export="ALL,MASTER_ROOT=$MASTER_ROOT" \
  "$@" \
  "$SCRIPT_DIRECTORY/extract.sbatch"
