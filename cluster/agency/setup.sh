#!/usr/bin/env bash
set -euo pipefail

# Create the run roots and sync the Python 3.12 environment.
#
# This half needs the **network** and no GPU, so it is safe on a login node --
# and it is also what `setup.sbatch` runs, for a cluster where the login node
# has no network or where interactive work is not allowed at all.
#
# The other half, proving that MuJoCo can render, needs a **GPU** and no
# network. It lives in `render_check.sh` and is submitted through
# `submit_filter.sh setup`.

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/common.sh"

mkdir -p \
  "$MEMORYMAZE_RUN_ROOT" \
  "$MEMORYMAZE_CORPUS_ROOT" \
  "$FILTER_RUN_ROOT" \
  "$SLURM_LOG_ROOT" \
  "$UV_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$MASTER_ROOT/tmp"

cd "$TB_REPO_ROOT"
uv sync --frozen --python "$MEMORYMAZE_PYTHON_VERSION" --extra memorymaze

echo
echo "Environment synced. Rendering is not verified yet -- it needs a GPU."
echo "Submit the render check with:"
echo
echo "    cluster/agency/submit_filter.sh setup --partition=<name> --time=00:15:00"
echo
