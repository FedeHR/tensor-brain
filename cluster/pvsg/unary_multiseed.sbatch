#!/usr/bin/env bash
#SBATCH --job-name=pvsg-unary-seeds0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-24%5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.harjes@campus.lmu.de

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/pvsg/common.sh"

CONDITIONS=(linear-probe fused-linear p-direct integral-none integral-p-sa)
# Fail with a readable message rather than an unbound-variable error when the submitted
# --array range does not match this script's task list.
_TASKS=("${CONDITIONS[@]}")
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "submit this script with --array; valid indices 0-$(( ${#_TASKS[@]} - 1 ))" >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID >= ${#_TASKS[@]} )); then
  echo "array index $SLURM_ARRAY_TASK_ID is out of range for $(basename "$0")" >&2
  echo "valid indices 0-$(( ${#_TASKS[@]} - 1 )): ${_TASKS[*]}" >&2
  exit 2
fi

TASK="$SLURM_ARRAY_TASK_ID"
CONDITION="${CONDITIONS[$((TASK % ${#CONDITIONS[@]}))]}"
SEED=$((TASK / ${#CONDITIONS[@]}))
RUN_NAME="unary-${CONDITION}-seed${SEED}"
OUTPUT_ROOT="$PVSG_RUN_ROOT/unary-multiseed/seed${SEED}"

export TMPDIR="$MASTER_ROOT/tmp/${SLURM_ARRAY_JOB_ID}_${TASK}"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$TMPDIR" "$OUTPUT_ROOT"

cd "$TB_REPO_ROOT"
bash cluster/pvsg/with_gpu_monitor.sh \
  uv run --frozen --no-sync python -m experiments.pvsg.object_experiment \
  --manifest-root "$PVSG_SECTION6_MANIFEST_ROOT" \
  --feature-root "$DINO_FEATURE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --condition "$CONDITION" \
  --semantic-condition hierarchy \
  --evolution qtb \
  --score-mode softplus-bias \
  --learning-rate 0.001 \
  --chunk-size 1024 \
  --seed "$SEED"
