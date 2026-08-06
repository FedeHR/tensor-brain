#!/usr/bin/env bash
#SBATCH --job-name=pvsg-pair-corrected
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-1%2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.harjes@campus.lmu.de

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/pvsg/common.sh"

VARIANT="${PAIR_TB_VARIANT:?submit with PAIR_TB_VARIANT=original or PAIR_TB_VARIANT=qtb}"
case "$VARIANT" in
  original)
    EVOLUTION=original
    SCORE_MODE=direct
    ;;
  qtb)
    EVOLUTION=qtb
    SCORE_MODE=softplus-bias
    ;;
  *)
    echo "PAIR_TB_VARIANT must be original or qtb, got: $VARIANT" >&2
    exit 2
    ;;
esac

CONDITIONS=(integral-none integral-p-sa)
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
RUN_NAME="pair-corrected-${VARIANT}-${CONDITION}-seed${PAIR_SEED:-0}"
OUTPUT_ROOT="$PVSG_RUN_ROOT/pair-corrected-${VARIANT}-seed${PAIR_SEED:-0}"

export TMPDIR="$MASTER_ROOT/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$TMPDIR" "$OUTPUT_ROOT"

cd "$TB_REPO_ROOT"
bash cluster/pvsg/with_gpu_monitor.sh \
  uv run --frozen --no-sync python -m experiments.pvsg.pair_experiment \
  --manifest-root "$PVSG_SECTION6_MANIFEST_ROOT" \
  --feature-root "$DINO_FEATURE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --condition "$CONDITION" \
  --evolution "$EVOLUTION" \
  --score-mode "$SCORE_MODE" \
  --learning-rate 0.0001 \
  --input-mapping-learning-rate 0.00001 \
  --validation-every 250 \
  --chunk-size 1024 \
  --seed "${PAIR_SEED:-0}" \
  --max-steps "${PAIR_MAX_STEPS:-10000}"
