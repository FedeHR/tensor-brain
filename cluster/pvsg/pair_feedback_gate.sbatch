#!/usr/bin/env bash
#SBATCH --job-name=pvsg-pair-feedback-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3%2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.harjes@campus.lmu.de

# Feedback-gate (beta) sweep on the category pathway.
#
# Across four rounds, index feedback moved predicate prediction by 0.0-0.3 pp. Every
# mechanical explanation has been excluded except injection magnitude:
#   - wrong candidate set        excluded, this pathway feeds back categories
#   - constant injection         excluded, direction cosine 0.06-0.24
#   - diffuse attention          excluded, max probability 0.81
#   - candidate-set distractors  excluded, unsupervised attention mass 0.3%
#
# What remains is scale. One injected embedding perturbs the state by ~4-7%, because
# the visual drive and the index embeddings are normalized independently:
# ||g(nu)|| : ||a_k|| is 27.7 : 1.0 at initialization and 34.0 : 2.4 at a selected
# checkpoint. QTB bounds beta by one, so beta cannot repair that ratio from inside the
# stated range; this sweep deliberately leaves it. beta = 8 puts the injection at
# roughly 30% of the visual drive.
#
# beta gates attention as well as the measurement injection. That ties the two
# magnitudes together, which is the point here: attention is the differentiable path
# these runs train on, so a measurement-only gate would have no training-time effect
# and the sweep would be four identical runs. Separating the two is the finer
# factorization if they ever need to move independently.
#
# beta=1..8 is done (runs/pair-feedback-gate-seed0): monotone improvement with no
# plateau, so the optimum lies beyond the swept range. Extend with PAIR_GATE_VALUES.
# PAIR_GATE_CONDITION=integral-p-sa repeats the ladder on the identity pathway, which
# is the paper's own P-SA mechanism and has only ever been measured at beta=1.
#
# The category pathway is swept rather than the identity one because its diagnostics
# are the strongest: near-one-hot attention over a group read out at 70-78% accuracy.
# If amplifying that changes nothing, amplifying the weaker identity pathway will not
# either. beta = 1 is retrained here rather than reused, so the ladder is exact.
#
# Each run also reports the QTB attention-then-measurement schedule at evaluation, so
# the sweep and the schedule question are answered by the same jobs. To read existing
# checkpoints under those schedules without retraining, use
# experiments.pvsg.pair_evaluate instead.

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/pvsg/common.sh"

# Override to extend the ladder without rerunning finished values, e.g.
#   PAIR_GATE_VALUES="16 32" sbatch --array=0-1 cluster/pvsg/pair_feedback_gate.sbatch
read -r -a GATES <<< "${PAIR_GATE_VALUES:-1 2 4 8}"
# Fail with a readable message rather than an unbound-variable error when the submitted
# --array range does not match this script's task list.
_TASKS=("${GATES[@]}")
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "submit this script with --array; valid indices 0-$(( ${#_TASKS[@]} - 1 ))" >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID >= ${#_TASKS[@]} )); then
  echo "array index $SLURM_ARRAY_TASK_ID is out of range for $(basename "$0")" >&2
  echo "valid indices 0-$(( ${#_TASKS[@]} - 1 )): ${_TASKS[*]}" >&2
  exit 2
fi

FEEDBACK_GATE="${GATES[$SLURM_ARRAY_TASK_ID]}"
CONDITION="${PAIR_GATE_CONDITION:-integral-cat-sa}"
RUN_NAME="pair-feedback-gate-${CONDITION}-beta${FEEDBACK_GATE}-seed${PAIR_SEED:-0}"
OUTPUT_ROOT="$PVSG_RUN_ROOT/pair-feedback-gate-seed${PAIR_SEED:-0}"

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
  --category-feedback-level source \
  --feedback-gate "$FEEDBACK_GATE" \
  --learning-rate 0.0001 \
  --input-mapping-learning-rate 0.00001 \
  --validation-every 250 \
  --chunk-size 1024 \
  --seed "${PAIR_SEED:-0}" \
  --max-steps "${PAIR_MAX_STEPS:-10000}"
