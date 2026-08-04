#!/usr/bin/env bash
#SBATCH --job-name=agency-memorymaze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-8%9

set -euo pipefail

# One array task per (condition, seed). Nine tasks: three conditions at three
# seeds. Each task is a single process, because Memory Maze's throughput comes
# from process count rather than from batching inside a process -- the adapter
# steps its environments in one Python loop and MuJoCo rendering is
# single-threaded. Two CPUs per task covers the stepping thread plus PyTorch.

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

uv run --frozen --no-sync --python "$MEMORYMAZE_PYTHON_VERSION" \
  python -m experiments.agency.memorymaze.run \
  --level "${MEMORYMAZE_LEVEL:-9x9}" \
  --condition "$CONDITION" \
  --seed "$SEED" \
  --output-root "$MEMORYMAZE_RUN_ROOT"
