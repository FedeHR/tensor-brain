#!/usr/bin/env bash
#SBATCH --job-name=agency-corpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --array=0-15%16

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

# 600 train / 100 probe / 100 test episodes at 1000 steps: about 9.8 GB.
# Tasks 0-11 render train, 12-13 probe, 14-15 test.
TASK="$SLURM_ARRAY_TASK_ID"
if (( TASK < 12 )); then
  SPLIT=train; CHUNK=$TASK;  EPISODES=50
elif (( TASK < 14 )); then
  SPLIT=probe; CHUNK=$((TASK - 12)); EPISODES=50
else
  SPLIT=test;  CHUNK=$((TASK - 14)); EPISODES=50
fi

# Distinct shard indices so the jobs never collide, and distinct seeds so they
# never render the same maze twice. Both are required; either alone is a bug.
SHARD_EPISODES=25
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

uv run --frozen --no-sync --python "$MEMORYMAZE_PYTHON_VERSION" \
  python -m experiments.agency.memorymaze.record \
  --root "$MEMORYMAZE_CORPUS_ROOT" \
  --split "$SPLIT" \
  --episodes "$EPISODES" \
  --shard-episodes "$SHARD_EPISODES" \
  --shard-offset "$SHARD_OFFSET" \
  --seed "$SEED"
