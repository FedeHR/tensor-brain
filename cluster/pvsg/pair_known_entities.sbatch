#!/usr/bin/env bash
#SBATCH --job-name=pvsg-pair-known-entities
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-4%3

# The known-entity pair protocol: the VRD-EX analogue the corrected runs never tested.
#
# Every condition trains on the first 45% of the frames of each training video and is
# reported on two evaluation sets from the same checkpoint:
#
#   development  heldout_video/development_pairs.jsonl  novel entities  -> VRD-E
#   blocked      blocked/evaluation_pairs.jsonl         same entities   -> VRD-EX
#
# Model selection stays on the novel-entity set, so the known-entity numbers are
# reported rather than selected for. Blocked is a stronger VRD-EX than the paper's:
# VRD-EX distorted copies of the training images (tb_original p.23), whereas blocked
# uses genuinely later frames behind a 10% temporal embargo.
#
# The five conditions isolate one mechanism each:
#   priors          frequency and category-pair floors. Blocked evaluates the same
#                   videos it trained on, so the prior shift must be measured before
#                   any blocked-versus-development gap can be attributed to the model.
#   union-only      DINO evidence with no index layer at all.
#   p-direct        the paper's P-Direct: no feedback and no dynamic context layer.
#   integral-none   dynamic context, no feedback. The one-factor control the paper
#                   never ran; p-direct minus this is the dynamic-context effect.
#   integral-p-sa   identity feedback, read back as both P-SA and P-Samp.
#
# Set PAIR_PROTOCOL=heldout_video to run the same conditions in the novel-entity
# regime only, which is what fills the corrected P-Direct cell that both existing
# arrays are missing.

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/pvsg/common.sh"

PROTOCOL="${PAIR_PROTOCOL:-blocked}"
CONDITIONS=(priors union-only p-direct integral-none integral-p-sa)
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"

# union-only keeps the learning rate of the complementarity study it is comparable
# with; the Tensor Brain family keeps the corrected pair runs' rates.
case "$CONDITION" in
  union-only)
    LEARNING_RATE=0.001
    ;;
  *)
    LEARNING_RATE=0.0001
    ;;
esac

RUN_NAME="pair-${PROTOCOL}-${CONDITION}-seed0"
OUTPUT_ROOT="$PVSG_RUN_ROOT/pair-${PROTOCOL}-seed0"

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
  --protocol "$PROTOCOL" \
  --condition "$CONDITION" \
  --evolution qtb \
  --score-mode softplus-bias \
  --learning-rate "$LEARNING_RATE" \
  --input-mapping-learning-rate 0.00001 \
  --validation-every 250 \
  --chunk-size 1024 \
  --seed 0
