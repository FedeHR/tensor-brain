#!/usr/bin/env bash

# One root owns every mutable cluster artifact; the repository remains code only.
# Mirrors cluster/pvsg/common.sh so both job families share cache and log layout.
MASTER_ROOT="${MASTER_ROOT:-/nfs/data8/harjes/MASTER}"
TB_REPO_ROOT="${TB_REPO_ROOT:-$MASTER_ROOT/tensor-brain}"
DIFFUSION_ROOT="${DIFFUSION_ROOT:-$MASTER_ROOT/diffusion}"
DIFFUSION_DATA_ROOT="${DIFFUSION_DATA_ROOT:-$DIFFUSION_ROOT/data}"
DIFFUSION_MODEL_CACHE="${DIFFUSION_MODEL_CACHE:-$DIFFUSION_ROOT/models}"
DIFFUSION_RUN_ROOT="${DIFFUSION_RUN_ROOT:-$MASTER_ROOT/runs/diffusion}"
SLURM_LOG_ROOT="${SLURM_LOG_ROOT:-$MASTER_ROOT/slurm/logs}"

export HF_HOME="$MASTER_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$MASTER_ROOT/cache/torch"
export UV_CACHE_DIR="$MASTER_ROOT/cache/uv"
export XDG_CACHE_HOME="$MASTER_ROOT/cache/xdg"
export PYTHONPATH="$TB_REPO_ROOT/src:$TB_REPO_ROOT"
export PYTHONUNBUFFERED=1

mkdir -p "$DIFFUSION_DATA_ROOT" "$DIFFUSION_MODEL_CACHE" "$DIFFUSION_RUN_ROOT"
