#!/usr/bin/env bash
#SBATCH --job-name=pvsg-pair-category-feedback
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-1%2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.harjes@campus.lmu.de

# Category index feedback on the corrected QTB pair Tensor Brain.
#
# The QTB family is fixed here deliberately: it beat the original-evolution family
# on every predicate metric (R@1 55.54 vs 50.52, KL 1.674 vs 1.864), and its
# softplus log-normalizer keeps the readout data-dependent instead of prior-driven.
#
# The no-feedback control is runs/pair-corrected-qtb-seed0/pair-corrected-qtb-integral-none-seed0
# and is reused rather than retrained, so this array is two jobs:
#   integral-cat-sa     category feedback only    (isolates the pathway)
#   integral-id-cat-sa  identity + category       (tests whether they compose)
# P-Samp is read back from each checkpoint at evaluation time.

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/pvsg/common.sh"

CONDITIONS=(integral-cat-sa integral-id-cat-sa)
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
CATEGORY_FEEDBACK_LEVEL="${PAIR_CATEGORY_FEEDBACK_LEVEL:-source}"
RUN_NAME="pair-category-feedback-${CONDITION}-seed${PAIR_SEED:-0}"
# PAIR_RUN_GROUP redirects the whole campaign into one directory, so a replicate
# cannot land on top of an earlier single-seed run. Run names already carry the
# protocol, condition and seed, so they stay unique inside a shared group.
OUTPUT_ROOT="$PVSG_RUN_ROOT/${PAIR_RUN_GROUP:-pair-category-feedback-seed${PAIR_SEED:-0}}"

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
  --evolution qtb \
  --score-mode softplus-bias \
  --category-feedback-level "$CATEGORY_FEEDBACK_LEVEL" \
  --learning-rate 0.0001 \
  --input-mapping-learning-rate 0.00001 \
  --validation-every 250 \
  --chunk-size 1024 \
  --seed "${PAIR_SEED:-0}" \
  --max-steps "${PAIR_MAX_STEPS:-10000}"
