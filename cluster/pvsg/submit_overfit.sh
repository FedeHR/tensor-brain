#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/common.sh"

if [[ -z "${PVSG_OVERFIT_RUN_NAME:-}" ]]; then
  echo "Set PVSG_OVERFIT_RUN_NAME to a unique, descriptive run name." >&2
  exit 2
fi
if [[ -e "$PVSG_RUN_ROOT/overfit/$PVSG_OVERFIT_RUN_NAME" ]]; then
  echo "Refusing to overwrite run: $PVSG_RUN_ROOT/overfit/$PVSG_OVERFIT_RUN_NAME" >&2
  exit 2
fi
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
PVSG_CODE_REVISION="$(git -C "$TB_REPO_ROOT" rev-parse HEAD)"
mkdir -p "$SLURM_LOG_ROOT" "$PVSG_RUN_ROOT/overfit"

sbatch \
  --output="$SLURM_LOG_ROOT/%x-%j.out" \
  --error="$SLURM_LOG_ROOT/%x-%j.err" \
  --export="ALL,MASTER_ROOT=$MASTER_ROOT,TB_REPO_ROOT=$TB_REPO_ROOT,PVSG_CODE_REVISION=$PVSG_CODE_REVISION" \
  "$@" \
  "$SCRIPT_DIRECTORY/overfit.sbatch"
