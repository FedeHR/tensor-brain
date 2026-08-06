#!/usr/bin/env bash
#SBATCH --job-name=pvsg-object-grid-chunked
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-11%12
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.harjes@campus.lmu.de

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/pvsg/common.sh"

EVOLUTIONS=(original qtb)
SCORES=(centered softplus-bias)
LEARNING_RATES=(0.0001 0.0003 0.001)
LR_TAGS=(1e-4 3e-4 1e-3)

TASK="$SLURM_ARRAY_TASK_ID"
EVOLUTION="${EVOLUTIONS[$((TASK / 6))]}"
SCORE="${SCORES[$(((TASK % 6) / 3))]}"
LR_INDEX=$((TASK % 3))
LEARNING_RATE="${LEARNING_RATES[$LR_INDEX]}"
RUN_NAME="object-chunked-${EVOLUTION}-${SCORE}-lr${LR_TAGS[$LR_INDEX]}-seed0"

export TMPDIR="$MASTER_ROOT/tmp/${SLURM_ARRAY_JOB_ID}_${TASK}"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$TMPDIR" "$PVSG_RUN_ROOT/object-grid-chunked"

cd "$TB_REPO_ROOT"
bash cluster/pvsg/with_gpu_monitor.sh \
  uv run --frozen --no-sync python -m experiments.pvsg.object_experiment \
  --manifest-root "$PVSG_SECTION6_MANIFEST_ROOT" \
  --feature-root "$DINO_FEATURE_ROOT" \
  --output-root "$PVSG_RUN_ROOT/object-grid-chunked" \
  --run-name "$RUN_NAME" \
  --evolution "$EVOLUTION" \
  --score-mode "$SCORE" \
  --chunk-size 1024 \
  --learning-rate "$LEARNING_RATE"
