#!/usr/bin/env bash
#SBATCH --job-name=agency-filter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=0-26%9

# `--mem` is host RAM. The corpus is read through mmap, so its pages are file
# cache rather than anonymous memory: under a cgroup they count toward the limit
# but are reclaimable, so a job is throttled rather than killed. 16G leaves room
# for the batch tensors and the reader.
#
# GPU memory is a separate budget and is not requested here. Activations
# dominate it and scale with batch_size x segment_steps (24 x 64 by default);
# if a card is too small, pass --batch-size or --segment-steps through to
# run_filter.py rather than changing the architecture.

set -euo pipefail

# The filter grid: nine conditions at three masking levels, one seed each.
#
# Unlike every other job in this directory, these tasks render nothing. They
# replay the recorded corpus, so they need no MuJoCo, no display and no EGL
# device -- only a GPU to train on. They are compute-bound rather than
# render-bound, which is the entire point of recording the corpus first.
#
# `%9` throttles to nine concurrent tasks; raise it if the partition allows.
# The corpus is read through mmap, so concurrent tasks share the page cache
# rather than each holding their own copy.

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/agency/common.sh"

CONDITIONS=(tb-none tb-raw tb-raw-alpha0 tb-corrected tb-softplus-a0 tb-soft tb-raw-noact tb-accumulator gru-control)
MASKS=(0.0 0.5 0.9)

TASK="$SLURM_ARRAY_TASK_ID"
CONDITION="${CONDITIONS[$((TASK / 3))]}"
MASK="${MASKS[$((TASK % 3))]}"
SEED="${FILTER_SEED:-0}"

export TMPDIR="$MASTER_ROOT/tmp/${SLURM_ARRAY_JOB_ID}_${TASK}"
mkdir -p "$TMPDIR" "$FILTER_RUN_ROOT"

cd "$TB_REPO_ROOT"
echo "task=$TASK condition=$CONDITION mask=$MASK seed=$SEED"
echo "code_revision=${AGENCY_CODE_REVISION:-unknown}"
echo "node=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-none}"

if [[ ! -f "$MEMORYMAZE_CORPUS_ROOT/train/metadata.json" ]]; then
  echo "No corpus at $MEMORYMAZE_CORPUS_ROOT." >&2
  echo "Record it first: cluster/agency/submit_corpus.sh" >&2
  exit 1
fi

uv run --frozen --no-sync --python "$MEMORYMAZE_PYTHON_VERSION" \
  python -m experiments.agency.memorymaze.run_filter \
  --corpus "$MEMORYMAZE_CORPUS_ROOT" \
  --condition "$CONDITION" \
  --mask "$MASK" \
  --seed "$SEED" \
  --output-root "$FILTER_RUN_ROOT"
