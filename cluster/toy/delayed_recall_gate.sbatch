#!/usr/bin/env bash
#SBATCH --job-name=toy-delayed-recall-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-5%6
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.harjes@campus.lmu.de

# Is index feedback harmful, or only mis-scaled?
#
# The first sweep found something sharper than a null. At silent recall and delay
# eight, the identical model solved the task in 5/5 seeds with the injection
# removed and collapsed to near chance with it at beta = 1. The same ordering
# appeared independently at probe recall with noise 0.6 and delay four. Index
# feedback is not neutral there; it costs.
#
# The mechanism that would explain it is structural. The injection writes the
# *current* item into q at every measurement, so across a delay the last thing
# written is the most recent distractor. The pathway is then a recency-biased
# overwrite of exactly the content that has to be maintained -- an interference
# channel rather than a memory one. If that is the explanation, accuracy should
# fall monotonically as beta rises, and this ladder is that test.
#
# beta = 0 is exactly the no-feedback condition and anchors the ladder against
# the earlier sweep. beta = 1 is the papers' value. The upper end is included
# because the PVSG category pathway settled near 25 and *helped* there, so the
# sign of this effect is not assumed to be universal.
#
# Read it as a dose-response curve, not as six separate cells. A monotone decline
# attributes the earlier collapse to the injection itself. A flat curve with a
# collapse only at beta = 1 would mean something else is going on and the first
# result should not be trusted.
#
# The delay and mode are fixed at the one cell where the earlier sweep proved the
# step budget sufficient: two of three arms trained to a 1.00 solve rate there,
# so a collapse cannot be blamed on under-training. Do not move them without
# re-establishing that.
#
# CPU-bound, like the main sweep:
#
#   SLURM_GRES=none SLURM_CPUS=4 SLURM_MEM=8G SLURM_TIME=4:00:00 \
#     cluster/pvsg/submit.sh --array=0-5%6 cluster/toy/delayed_recall_gate.sbatch

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/pvsg/common.sh"

read -r -a GATES <<< "${TOY_GATE_VALUES:-0 0.25 0.5 1 2 4}"
DELAY="${TOY_DELAY:-8}"
RECALL_MODE="${TOY_RECALL_MODE:-silent}"
VIEW_NOISE="${TOY_VIEW_NOISE:-0}"
EVOLUTION="${TOY_EVOLUTION:-qtb-relu}"
TRAINING_STEPS="${TOY_MAX_STEPS:-12000}"
read -r -a SEEDS <<< "${TOY_SEEDS:-0 1 2 3 4}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "submit this script with --array; valid indices 0-$(( ${#GATES[@]} - 1 ))" >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID >= ${#GATES[@]} )); then
  echo "array index $SLURM_ARRAY_TASK_ID is out of range for $(basename "$0")" >&2
  echo "valid indices 0-$(( ${#GATES[@]} - 1 )): ${GATES[*]}" >&2
  exit 2
fi

GATE="${GATES[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_ROOT="${MASTER_ROOT}/runs/toy/${TOY_RUN_GROUP:-delayed-recall-gate}"
# One file per array task; see the note in delayed_recall.sbatch.
STEM="${RECALL_MODE}-noise${VIEW_NOISE}-delay${DELAY}-${EVOLUTION}"
RESULTS_PATH="$OUTPUT_ROOT/${STEM}-beta${GATE}.jsonl"

export TMPDIR="$MASTER_ROOT/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$OUTPUT_ROOT"

echo "beta=$GATE delay=$DELAY mode=$RECALL_MODE noise=$VIEW_NOISE" \
     "evolution=$EVOLUTION steps=$TRAINING_STEPS seeds=${SEEDS[*]}"

cd "$TB_REPO_ROOT"
uv run --frozen --no-sync python -m experiments.delayed_recall \
  --condition feedback \
  --delay "$DELAY" \
  --recall-mode "$RECALL_MODE" \
  --view-noise "$VIEW_NOISE" \
  --feedback-gate "$GATE" \
  --evolution "$EVOLUTION" \
  --training-steps "$TRAINING_STEPS" \
  --seeds "${SEEDS[@]}" \
  --output "$RESULTS_PATH"
