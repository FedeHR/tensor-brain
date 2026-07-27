#!/usr/bin/env bash
#SBATCH --job-name=pvsg-dinov3
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/common.sh"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This job must be submitted as a Slurm array." >&2
  exit 2
fi

JOB_TMP="$MASTER_ROOT/tmp/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
mkdir -p "$JOB_TMP"
export TMPDIR="$JOB_TMP"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$TB_REPO_ROOT"
uv run --frozen --no-sync python -m experiments.pvsg.extract \
  --manifest "$PVSG_MANIFEST_ROOT/videos.jsonl" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --dataset-root "$PVSG_DATASET_ROOT" \
  --output-root "$DINO_FEATURE_ROOT" \
  --device cuda \
  --batch-size "${DINO_BATCH_SIZE:-8}" \
  --long-edge 448
