#!/usr/bin/env bash
#SBATCH --job-name=agency-memorymaze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%9

set -euo pipefail

# One array task per (condition, seed). Nine tasks: three conditions at three
# seeds. Each task is a single process, because Memory Maze's throughput comes
# from process count rather than from batching inside a process -- the adapter
# steps its environments in one Python loop and MuJoCo rendering is
# single-threaded. Two CPUs per task covers the stepping thread plus PyTorch.
#
# The GPU request is not optional: MUJOCO_GL=egl renders against a GPU, so a
# task without one has no EGL device to open and cannot produce an observation.
# Override `--gres` at submission time if this cluster names its GPUs otherwise;
# on a CPU-only partition, submit with MUJOCO_GL=osmesa instead and expect it to
# be far slower.

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/agency/common.sh"

CONDITIONS=(tb-full gru-control lstm-control)
SEEDS=(0 1 2)

TASK="$SLURM_ARRAY_TASK_ID"
CONDITION="${CONDITIONS[$((TASK / 3))]}"
SEED="${SEEDS[$((TASK % 3))]}"

export TMPDIR="$MASTER_ROOT/tmp/${SLURM_ARRAY_JOB_ID}_${TASK}"
mkdir -p "$TMPDIR" "$MEMORYMAZE_RUN_ROOT"

cd "$TB_REPO_ROOT"
echo "task=$TASK condition=$CONDITION seed=$SEED MUJOCO_GL=$MUJOCO_GL"
echo "code_revision=${AGENCY_CODE_REVISION:-unknown}"
echo "node=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-none}"

# MuJoCo selects its EGL device by index over *all* GPUs on the machine and does
# not consult CUDA_VISIBLE_DEVICES, so on a multi-GPU node it would happily
# render on a card Slurm did not allocate. Point it at the first device we were
# actually given.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$MUJOCO_GL" == "egl" ]]; then
  export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"
  echo "MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
fi

# Preflight: build one environment and render one frame. This costs a few
# seconds and converts the expensive failure -- a task that occupies a slot for
# hours without ever producing an observation -- into an immediate, legible one.
if ! uv run --frozen --no-sync --python "$MEMORYMAZE_PYTHON_VERSION" python -c "
from experiments.agency.memorymaze.env import VectorMemoryMaze
environment = VectorMemoryMaze(1, seed=0)
assert environment.observation().shape == (1, environment.observation_dim)
environment.close()
print('preflight: rendering OK')
"; then
  echo "preflight FAILED: no usable rendering backend (MUJOCO_GL=$MUJOCO_GL)." >&2
  echo "On a node without a usable EGL device, resubmit with MUJOCO_GL=osmesa." >&2
  exit 1
fi

uv run --frozen --no-sync --python "$MEMORYMAZE_PYTHON_VERSION" \
  python -m experiments.agency.memorymaze.run \
  --level "${MEMORYMAZE_LEVEL:-9x9}" \
  --condition "$CONDITION" \
  --seed "$SEED" \
  --output-root "$MEMORYMAZE_RUN_ROOT"
