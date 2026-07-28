#!/usr/bin/env bash
#SBATCH --job-name=pvsg-dinov3
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

# Slurm may execute a staged copy of this file from /var/lib/slurm/... .  Resolve
# repository files through the exported cluster root rather than through BASH_SOURCE.
REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
SCRIPT_DIRECTORY="$REPOSITORY_ROOT/cluster/pvsg"
source "$SCRIPT_DIRECTORY/common.sh"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This job must be submitted as a Slurm array." >&2
  exit 2
fi
if [[ ! -d "$TB_REPO_ROOT" ]]; then
  echo "Tensor Brain repository is missing: $TB_REPO_ROOT" >&2
  exit 2
fi
if [[ ! -x "$SCRIPT_DIRECTORY/with_gpu_monitor.sh" ]]; then
  echo "GPU monitor script is missing or not executable: $SCRIPT_DIRECTORY/with_gpu_monitor.sh" >&2
  exit 2
fi
if [[ ! -s "$PVSG_MANIFEST_ROOT/videos.jsonl" ]]; then
  echo "PVSG extraction manifest is missing: $PVSG_MANIFEST_ROOT/videos.jsonl" >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is unavailable in the Slurm worker PATH" >&2
  exit 2
fi

JOB_TMP="$MASTER_ROOT/tmp/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
mkdir -p "$JOB_TMP"
export TMPDIR="$JOB_TMP"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$TB_REPO_ROOT"
bash "$SCRIPT_DIRECTORY/with_gpu_monitor.sh" \
  uv run --frozen --no-sync python -m experiments.pvsg.extract \
  --manifest "$PVSG_MANIFEST_ROOT/videos.jsonl" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --dataset-root "$PVSG_DATASET_ROOT" \
  --output-root "$DINO_FEATURE_ROOT" \
  --device cuda \
  --batch-size "${DINO_BATCH_SIZE:-4}" \
  --long-edge 448
