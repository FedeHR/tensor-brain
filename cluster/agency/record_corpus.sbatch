#!/usr/bin/env bash
#SBATCH --job-name=agency-corpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --array=0-15%16

# `--mem` is host RAM, and 8G is generous: the peak is one shard held before it
# is written (24 episodes x 1000 steps x 12,288 bytes = 295 MB, briefly doubled
# by the concatenate) plus MuJoCo and PyTorch. Any value passed to
# `submit_filter.sh` overrides this, since sbatch command-line arguments win
# over #SBATCH directives.

set -euo pipefail

# Render the offline corpus once. Every filter run replays it, so this is the
# only rendering the study pays for, and it is why the study is affordable at
# all: an on-policy method re-renders on every run.
#
# Sixteen tasks over disjoint shard ranges. Rendering does not parallelise inside
# a process -- the adapter steps its environments in one Python loop and MuJoCo
# is single-threaded -- so throughput is entirely a function of process count.
# At roughly 180 frames/s per task, 800k frames takes about ten minutes.
#
# The GPU request is not optional: MUJOCO_GL=egl renders against a GPU. See
# memorymaze.sbatch for the same caveat and the osmesa fallback.

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/agency/common.sh"

# 576 train / 96 probe / 96 test episodes at 1000 steps: about 9.4 GB.
# Tasks 0-11 render train, 12-13 probe, 14-15 test.
#
# 48 episodes per task and 24 per shard, because `CorpusConfig` requires both to
# be multiples of the batch width (8 environments) -- episodes are produced a
# full batch at a time, so a count that is not a multiple would leave a partial
# batch with nowhere to go.
EPISODES=48
SHARD_EPISODES=24

TASK="$SLURM_ARRAY_TASK_ID"
if (( TASK < 12 )); then
  SPLIT=train; CHUNK=$TASK
elif (( TASK < 14 )); then
  SPLIT=probe; CHUNK=$((TASK - 12))
else
  SPLIT=test;  CHUNK=$((TASK - 14))
fi

# Distinct shard indices so the jobs never collide, and distinct seeds so they
# never render the same maze twice. Both are required; either alone is a bug.
SHARD_OFFSET=$((CHUNK * EPISODES / SHARD_EPISODES))
case "$SPLIT" in
  train) SEED_BASE=0 ;;
  probe) SEED_BASE=500000 ;;
  test)  SEED_BASE=900000 ;;
esac
SEED=$((SEED_BASE + CHUNK * 10000))

export TMPDIR="$MASTER_ROOT/tmp/${SLURM_ARRAY_JOB_ID}_${TASK}"
mkdir -p "$TMPDIR" "$MEMORYMAZE_CORPUS_ROOT"

cd "$TB_REPO_ROOT"
echo "task=$TASK split=$SPLIT chunk=$CHUNK episodes=$EPISODES seed=$SEED shard_offset=$SHARD_OFFSET"
echo "code_revision=${AGENCY_CODE_REVISION:-unknown}"
echo "node=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-none} MUJOCO_GL=$MUJOCO_GL"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$MUJOCO_GL" == "egl" ]]; then
  export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"
fi

# Preflight. A few seconds, and it turns "this task occupied a slot for an hour
# without ever producing a frame" into an immediate, legible failure.
if ! cluster/agency/render_check.sh; then
  echo "preflight FAILED: no usable rendering backend (MUJOCO_GL=$MUJOCO_GL)." >&2
  echo "On a node without a usable EGL device, resubmit with MUJOCO_GL=osmesa." >&2
  exit 1
fi

uv run --frozen --no-sync --python "$MEMORYMAZE_PYTHON_VERSION" \
  python -m experiments.agency.memorymaze.record \
  --root "$MEMORYMAZE_CORPUS_ROOT" \
  --split "$SPLIT" \
  --episodes "$EPISODES" \
  --shard-episodes "$SHARD_EPISODES" \
  --shard-offset "$SHARD_OFFSET" \
  --seed "$SEED"
