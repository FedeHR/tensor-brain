#!/usr/bin/env bash
set -euo pipefail

# Prepare the cluster for the Memory Maze study and prove the rendering backend
# works *before* an array job discovers otherwise nine times in parallel.

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/common.sh"

mkdir -p \
  "$MEMORYMAZE_RUN_ROOT" \
  "$SLURM_LOG_ROOT" \
  "$UV_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$MASTER_ROOT/tmp"

cd "$TB_REPO_ROOT"
uv sync --frozen --python "$MEMORYMAZE_PYTHON_VERSION" --extra memorymaze

# The smoke test: build one environment, take one step, and confirm the
# observation is the 64x64 RGB image the encoder expects and that ground truth
# is present. Run this on a *compute* node, not the login node -- the login node
# usually has no GPU, so EGL will fail there even when the batch nodes are fine.
uv run --frozen --no-sync --python "$MEMORYMAZE_PYTHON_VERSION" python - <<'PYTHON'
import os
import torch

from experiments.agency.memorymaze.env import VectorMemoryMaze

environment = VectorMemoryMaze(1, seed=0)
observation = environment.observation()
truth = environment.ground_truth()
environment.step(torch.zeros(1, dtype=torch.long))
print(f"MUJOCO_GL={os.environ['MUJOCO_GL']}")
print(f"observation={tuple(observation.shape)} expected=(1, {environment.observation_dim})")
assert observation.shape == (1, environment.observation_dim)
assert {"agent_pos", "targets_pos", "target_slot"} <= set(truth)
print("ground truth:", {key: tuple(value.shape) for key, value in truth.items()})
environment.close()
print("memory maze OK")
PYTHON
