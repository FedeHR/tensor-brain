#!/usr/bin/env bash
#SBATCH --job-name=pvsg-pair-complementarity-seed0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3%4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.harjes@campus.lmu.de

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/pvsg/common.sh"

CONDITIONS=(
  category-only
  union-only
  union-category-oracle
  union-category-predicted
)
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

CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="pair-${CONDITION}-seed${PAIR_SEED:-0}"

export TMPDIR="$MASTER_ROOT/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$TMPDIR" "$PVSG_RUN_ROOT/pair-complementarity-seed0"

cd "$TB_REPO_ROOT"
bash cluster/pvsg/with_gpu_monitor.sh \
  uv run --frozen --no-sync python -m experiments.pvsg.pair_experiment \
  --manifest-root "$PVSG_SECTION6_MANIFEST_ROOT" \
  --feature-root "$DINO_FEATURE_ROOT" \
  --output-root "$PVSG_RUN_ROOT/pair-complementarity-seed0" \
  --run-name "$RUN_NAME" \
  --condition "$CONDITION" \
  --learning-rate 0.001 \
  --chunk-size 1024 \
  --seed "${PAIR_SEED:-0}" \
  --max-steps "${PAIR_MAX_STEPS:-10000}"
