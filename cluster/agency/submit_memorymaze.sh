#!/usr/bin/env bash
set -euo pipefail

# Submit the Memory Maze array job. Extra arguments are passed through to
# `sbatch`, which is where partition, time limit and any GPU request belong:
#
#   cluster/agency/submit_memorymaze.sh --partition=<name> --time=04:00:00
#
# Run `cluster/agency/setup.sh` on a compute node first: it syncs the 3.12
# environment and proves the rendering backend works, which is the failure this
# study is most likely to hit and the most wasteful one to discover nine times
# in parallel.

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/common.sh"

# The run artifacts are only interpretable against the code that produced them,
# so refuse to submit anything that is not committed -- the same rule the PVSG
# submitters enforce.
if ! git -C "$TB_REPO_ROOT" diff --quiet -- experiments src cluster pyproject.toml uv.lock; then
  echo "Tracked experiment code has uncommitted changes; commit it before submission." >&2
  exit 2
fi
if ! git -C "$TB_REPO_ROOT" diff --cached --quiet -- experiments src cluster pyproject.toml uv.lock; then
  echo "Experiment code has staged but uncommitted changes; commit it before submission." >&2
  exit 2
fi
if [[ -n "$(git -C "$TB_REPO_ROOT" ls-files --others --exclude-standard -- experiments src cluster)" ]]; then
  echo "Experiment code contains untracked files; commit them before submission." >&2
  exit 2
fi

AGENCY_CODE_REVISION="$(git -C "$TB_REPO_ROOT" rev-parse HEAD)"
mkdir -p "$SLURM_LOG_ROOT" "$MEMORYMAZE_RUN_ROOT"

sbatch \
  --output="$SLURM_LOG_ROOT/%x-%A_%a.out" \
  --error="$SLURM_LOG_ROOT/%x-%A_%a.err" \
  --export="ALL,MASTER_ROOT=$MASTER_ROOT,TB_REPO_ROOT=$TB_REPO_ROOT,MUJOCO_GL=$MUJOCO_GL,AGENCY_CODE_REVISION=$AGENCY_CODE_REVISION" \
  "$@" \
  "$SCRIPT_DIRECTORY/memorymaze.sbatch"
