#!/usr/bin/env bash
#SBATCH --job-name=pvsg-pair-oracle-residual-seed0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.harjes@campus.lmu.de

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/pvsg/common.sh"

RUN_NAME="pair-union-category-oracle-residual-seed${PAIR_SEED:-0}"

export TMPDIR="$MASTER_ROOT/tmp/${SLURM_JOB_ID}_oracle_residual"
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
  --condition union-category-oracle \
  --zero-initialize-union \
  --learning-rate 0.0001 \
  --max-steps 2000 \
  --validation-every 100 \
  --log-every 100 \
  --chunk-size 1024 \
  --seed "${PAIR_SEED:-0}" \
  --max-steps "${PAIR_MAX_STEPS:-10000}"
