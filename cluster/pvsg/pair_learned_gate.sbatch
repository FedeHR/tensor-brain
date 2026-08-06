#!/usr/bin/env bash
#SBATCH --job-name=pvsg-pair-learned-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-1%2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.harjes@campus.lmu.de

# Learn beta instead of fixing it, and let the model report the injection magnitude it
# prefers.
#
# The fixed sweep found monotone improvement from beta = 1 to 8 with no plateau, so the
# optimum is outside the swept range and picking a value by grid search would be both
# expensive and arbitrary. A learned gate locates it directly, and its trajectory is
# recorded every logged step so the approach can be read rather than inferred.
#
# beta is softplus-parameterized, so it stays positive and spans QTB's stated 0-to-1
# range together with the extension the sweep explored. It initializes at exactly the
# paper's beta = 1, which makes step zero identical to the corrected runs. It gets its
# own optimizer group: at the main rate one scalar could not traverse the range within
# the step budget.
#
# Two pathways, because the identity one is the paper's own mechanism and has only
# been measured at beta = 1, where it was worth +3 assignments of 67,134.
#
# PAIR_PROTOCOL=blocked is the sharp test. On heldout_video the learned identity gate
# collapsed to 0.002 - but there identity *cannot* be correct, since PVSG identities are
# video-scoped. On blocked the same index is recoverable at 44.5% pair-exact, so whether
# the gate still collapses separates "this pathway is useless" from "this pathway needs
# a retrievable index", which is the paper's own VRD-E versus VRD-EX contrast.
# Note blocked trains on 168,621 pairs against heldout_video's 367,132, so compare the
# learned gate values qualitatively, not the metrics across protocols.

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/pvsg/common.sh"

PROTOCOL="${PAIR_PROTOCOL:-heldout_video}"
CONDITIONS=(integral-cat-sa integral-p-sa)
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
GATE_LEARNING_RATE="${PAIR_GATE_LEARNING_RATE:-0.01}"
RUN_NAME="pair-learned-gate-${PROTOCOL}-${CONDITION}-seed0"
OUTPUT_ROOT="$PVSG_RUN_ROOT/pair-learned-gate-${PROTOCOL}-seed0"

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
  --protocol "$PROTOCOL" \
  --evolution qtb \
  --score-mode softplus-bias \
  --category-feedback-level source \
  --learn-feedback-gate \
  --feedback-gate-learning-rate "$GATE_LEARNING_RATE" \
  --learning-rate 0.0001 \
  --input-mapping-learning-rate 0.00001 \
  --validation-every 250 \
  --chunk-size 1024 \
  --seed 0
