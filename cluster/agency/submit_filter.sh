#!/usr/bin/env bash
set -euo pipefail

# Submit one stage of the offline filter study. Extra arguments pass through to
# `sbatch`, which is where partition and time limit belong:
#
#   cluster/agency/submit_filter.sh corpus --partition=<name> --time=01:00:00
#   cluster/agency/submit_filter.sh grid   --partition=<name> --time=04:00:00
#
# `corpus` renders the offline trajectories once; `grid` fits and probes the
# nine conditions at three masking levels over that corpus. The second requires
# the first to have finished.

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/common.sh"

STAGE="${1:-}"
case "$STAGE" in
  corpus) BATCH="$SCRIPT_DIRECTORY/record_corpus.sbatch" ;;
  grid)   BATCH="$SCRIPT_DIRECTORY/filter.sbatch" ;;
  *) echo "usage: $0 {corpus|grid} [sbatch arguments...]" >&2; exit 2 ;;
esac
shift

# Run artifacts are only interpretable against the code that produced them.
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

if [[ "$STAGE" == "grid" && ! -f "$MEMORYMAZE_CORPUS_ROOT/train/metadata.json" ]]; then
  echo "No corpus at $MEMORYMAZE_CORPUS_ROOT; run '$0 corpus' first." >&2
  exit 2
fi

AGENCY_CODE_REVISION="$(git -C "$TB_REPO_ROOT" rev-parse HEAD)"
mkdir -p "$SLURM_LOG_ROOT" "$MEMORYMAZE_CORPUS_ROOT" "$FILTER_RUN_ROOT"

sbatch \
  --output="$SLURM_LOG_ROOT/%x-%A_%a.out" \
  --error="$SLURM_LOG_ROOT/%x-%A_%a.err" \
  --export="ALL,MASTER_ROOT=$MASTER_ROOT,TB_REPO_ROOT=$TB_REPO_ROOT,MUJOCO_GL=$MUJOCO_GL,MEMORYMAZE_CORPUS_ROOT=$MEMORYMAZE_CORPUS_ROOT,FILTER_RUN_ROOT=$FILTER_RUN_ROOT,AGENCY_CODE_REVISION=$AGENCY_CODE_REVISION" \
  "$@" \
  "$BATCH"
