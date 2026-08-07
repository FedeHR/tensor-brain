#!/usr/bin/env bash
#SBATCH --job-name=toy-delayed-recall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-11%6
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.harjes@campus.lmu.de

# The delayed cued-recall gate, swept over delay for all three conditions.
#
# What this is for. The PVSG pair experiments found index feedback worth +0.09 pp
# and could not say whether that is a property of the mechanism or of the setting.
# The setting made feedback redundant: the identity was a deterministic function of
# the crop in front of the model, so the injected column carried nothing the input
# did not. This gate removes that confound in a place where every variable is
# controlled, and asks whether feedback is worth more than the analog trace the
# same observation already left in q.
#
# Two things make the question answerable here and not in the earlier toy:
#
#   TOY_VIEW_NOISE > 0    every presentation is a noisy view of an item prototype,
#                         so a_k is a prototype accumulated across occurrences
#                         rather than a re-encoding of one input. This is the
#                         property original section 6 relies on for VRD-EX.
#   TOY_RECALL_MODE=probe the recall step presents evidence that is equidistant
#                         between the cue and a lure, so bottom-up narrows the
#                         answer to two candidates and cannot choose. A model
#                         reading only the probe scores 0.5; that is the floor,
#                         and it is reported as `chance` in the output.
#
# Read the delay axis, not any single cell. The claim that index feedback buys
# anything is a claim about the *slope*: a discrete symbol re-injected on every
# measurement should degrade more slowly than a continuous trace passed through
# the evolution operator. Delays 1 and 2 are saturated for every condition and
# say nothing. If feedback and no-feedback fall together across the whole range,
# that is a clean negative about the mechanism itself and the PVSG streaming
# experiment should not be built.
#
# This workload is CPU-bound: the models are tiny and the forward pass is a Python
# loop over sequential steps, so a GPU buys very little and the array width is what
# matters. Prefer
#
#   SLURM_GRES=none SLURM_CPUS=4 SLURM_MEM=8G SLURM_TIME=4:00:00 \
#     cluster/pvsg/submit.sh --array=0-11%6 cluster/toy/delayed_recall.sbatch
#
# Run the silent variant for contrast with TOY_RECALL_MODE=silent TOY_VIEW_NOISE=0.
# Every cell appends one JSON line to the shared results file, so the two sweeps
# can share a run group and be aggregated together.

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/pvsg/common.sh"

CONDITIONS=(feedback no-feedback gru)
read -r -a DELAYS <<< "${TOY_DELAYS:-4 8 16 32}"
RECALL_MODE="${TOY_RECALL_MODE:-probe}"
VIEW_NOISE="${TOY_VIEW_NOISE:-0.3}"
EVOLUTION="${TOY_EVOLUTION:-qtb-relu}"
TRAINING_STEPS="${TOY_MAX_STEPS:-12000}"
read -r -a SEEDS <<< "${TOY_SEEDS:-0 1 2 3 4}"

TASK_COUNT=$(( ${#CONDITIONS[@]} * ${#DELAYS[@]} ))
# Fail with a readable message rather than an unbound-variable error when the
# submitted --array range does not match this script's task list.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "submit this script with --array; valid indices 0-$(( TASK_COUNT - 1 ))" >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID >= TASK_COUNT )); then
  echo "array index $SLURM_ARRAY_TASK_ID is out of range for $(basename "$0")" >&2
  echo "valid indices 0-$(( TASK_COUNT - 1 )): ${#CONDITIONS[@]} conditions x ${#DELAYS[@]} delays" >&2
  exit 2
fi

CONDITION="${CONDITIONS[$(( SLURM_ARRAY_TASK_ID % ${#CONDITIONS[@]} ))]}"
DELAY="${DELAYS[$(( SLURM_ARRAY_TASK_ID / ${#CONDITIONS[@]} ))]}"

OUTPUT_ROOT="${MASTER_ROOT}/runs/toy/${TOY_RUN_GROUP:-delayed-recall}"
# One file per array task. Concurrent appends to a shared file are not atomic on
# NFS, so a throttle wider than one could silently drop a finished cell and leave
# a gap indistinguishable from a job that never ran. Reassemble with
#   cat "$OUTPUT_ROOT"/<stem>-task*.jsonl
STEM="${RECALL_MODE}-noise${VIEW_NOISE}-${EVOLUTION}"
RESULTS_PATH="$OUTPUT_ROOT/${STEM}-task${SLURM_ARRAY_TASK_ID}.jsonl"

export TMPDIR="$MASTER_ROOT/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$OUTPUT_ROOT"

echo "condition=$CONDITION delay=$DELAY mode=$RECALL_MODE noise=$VIEW_NOISE" \
     "evolution=$EVOLUTION steps=$TRAINING_STEPS seeds=${SEEDS[*]}"

cd "$TB_REPO_ROOT"
uv run --frozen --no-sync python -m experiments.delayed_recall \
  --condition "$CONDITION" \
  --delay "$DELAY" \
  --recall-mode "$RECALL_MODE" \
  --view-noise "$VIEW_NOISE" \
  --evolution "$EVOLUTION" \
  --training-steps "$TRAINING_STEPS" \
  --seeds "${SEEDS[@]}" \
  --output "$RESULTS_PATH"
