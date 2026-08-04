#!/usr/bin/env bash
#SBATCH --job-name=pvsg-pair-seed0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-6%7

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/pvsg/common.sh"

CONDITIONS=(
  priors
  linear-probe
  fused-linear
  flat-fusion
  p-direct
  integral-none
  integral-p-sa
)
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="pair-${CONDITION}-seed0"

export TMPDIR="$MASTER_ROOT/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$TMPDIR" "$PVSG_RUN_ROOT/pair-seed0"

cd "$TB_REPO_ROOT"
bash cluster/pvsg/with_gpu_monitor.sh \
  uv run --frozen --no-sync python -m experiments.pvsg.pair_experiment \
  --manifest-root "$PVSG_SECTION6_MANIFEST_ROOT" \
  --feature-root "$DINO_FEATURE_ROOT" \
  --output-root "$PVSG_RUN_ROOT/pair-seed0" \
  --run-name "$RUN_NAME" \
  --condition "$CONDITION" \
  --evolution qtb \
  --score-mode softplus-bias \
  --learning-rate 0.001 \
  --chunk-size 1024 \
  --seed 0
