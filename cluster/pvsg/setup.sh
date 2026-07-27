#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/common.sh"

mkdir -p \
  "$PVSG_ARCHIVES_ROOT" \
  "$PVSG_ROOT/staging" \
  "$PVSG_MANIFEST_ROOT" \
  "$DINO_FEATURE_ROOT" \
  "$PVSG_RUN_ROOT" \
  "$SLURM_LOG_ROOT" \
  "$HF_HOME" \
  "$TORCH_HOME" \
  "$UV_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$MASTER_ROOT/tmp"

cd "$TB_REPO_ROOT"
uv sync --frozen --extra pvsg

uv run --frozen --no-sync hf download Jingkang/PVSG \
  Ego4D/ego4d_masks.zip \
  Ego4D/ego4d_videos.zip \
  EpicKitchen/epic_kitchen_masks.zip \
  EpicKitchen/epic_kitchen_videos.zip \
  VidOR/vidor_masks.zip \
  VidOR/vidor_videos.zip \
  pvsg.json \
  --repo-type dataset \
  --revision 7e5f1ec9fd8f323182e84d990819854bb72da478 \
  --local-dir "$PVSG_ARCHIVES_ROOT"

uv run --frozen --no-sync hf download \
  facebook/dinov3-vitb16-pretrain-lvd1689m \
  --revision "$DINO_MODEL_REVISION"

uv run --frozen --no-sync python -m experiments.pvsg.prepare \
  --pvsg-root "$PVSG_ROOT"
