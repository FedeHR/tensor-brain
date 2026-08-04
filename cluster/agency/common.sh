#!/usr/bin/env bash

# One root owns every mutable cluster artifact; the repository remains code only.
MASTER_ROOT="${MASTER_ROOT:-/nfs/data8/harjes/MASTER}"
TB_REPO_ROOT="${TB_REPO_ROOT:-$MASTER_ROOT/tensor-brain}"
AGENCY_RUN_ROOT="${AGENCY_RUN_ROOT:-$MASTER_ROOT/runs/agency}"
MEMORYMAZE_RUN_ROOT="${MEMORYMAZE_RUN_ROOT:-$AGENCY_RUN_ROOT/memorymaze}"
SLURM_LOG_ROOT="${SLURM_LOG_ROOT:-$MASTER_ROOT/slurm/logs}"

# Memory Maze needs Python 3.12: `labmaze` ships no 3.13 wheel.
MEMORYMAZE_PYTHON_VERSION="${MEMORYMAZE_PYTHON_VERSION:-3.12}"

# MuJoCo's rendering backend is the one setting that does not travel between
# macOS and a batch node. `egl` renders offscreen against the GPU and is what a
# headless node wants; set MUJOCO_GL=osmesa before submitting if a node has no
# usable EGL device, at a substantial cost in speed.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

export UV_CACHE_DIR="$MASTER_ROOT/cache/uv"
export XDG_CACHE_HOME="$MASTER_ROOT/cache/xdg"
export PYTHONUNBUFFERED=1

# Each array task is one single-threaded process; without this, every task
# spawns a full thread pool and they contend rather than run.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
