#!/usr/bin/env bash
set -euo pipefail

# Build one Memory Maze environment and render one frame.
#
# This needs a GPU, because MUJOCO_GL=egl renders against one. It is separated
# from `setup.sh` for exactly that reason: package installation needs the
# network and no GPU, rendering needs a GPU and no network, and on most clusters
# those are two different machines. Keeping them apart means neither has to run
# somewhere it cannot.
#
# The check is worth its few seconds: a broken rendering backend is the failure
# this study is most likely to hit, and the most wasteful one to discover
# sixteen times in parallel after the array has already started.

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/common.sh"

cd "$TB_REPO_ROOT"

# MuJoCo picks its EGL device by index over *all* GPUs on the machine and does
# not consult CUDA_VISIBLE_DEVICES, so on a multi-GPU node it would happily
# render on a card Slurm did not allocate.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$MUJOCO_GL" == "egl" ]]; then
  export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"
fi

echo "node=$(hostname) MUJOCO_GL=$MUJOCO_GL cuda_visible=${CUDA_VISIBLE_DEVICES:-none}"

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
