#!/usr/bin/env bash
#SBATCH --job-name=pvsg-overfit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.harjes@campus.lmu.de

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
SCRIPT_DIRECTORY="$REPOSITORY_ROOT/cluster/pvsg"
source "$SCRIPT_DIRECTORY/common.sh"

if [[ -z "${PVSG_OVERFIT_RUN_NAME:-}" ]]; then
  echo "PVSG_OVERFIT_RUN_NAME is required." >&2
  exit 2
fi
if [[ ! -s "$PVSG_SECTION6_MANIFEST_ROOT/heldout_video/train_pairs.jsonl" ]]; then
  echo "PVSG training-pair manifest is missing: $PVSG_SECTION6_MANIFEST_ROOT" >&2
  exit 2
fi
if [[ ! -s "$PVSG_SECTION6_MANIFEST_ROOT/ontology.json" ]]; then
  echo "PVSG ontology is missing: $PVSG_SECTION6_MANIFEST_ROOT/ontology.json" >&2
  exit 2
fi
if [[ ! -d "$DINO_FEATURE_ROOT/videos" ]]; then
  echo "DINO feature snapshot is missing: $DINO_FEATURE_ROOT" >&2
  exit 2
fi
if [[ -e "$PVSG_RUN_ROOT/overfit/$PVSG_OVERFIT_RUN_NAME" ]]; then
  echo "Run already exists: $PVSG_RUN_ROOT/overfit/$PVSG_OVERFIT_RUN_NAME" >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is unavailable in the Slurm worker PATH" >&2
  exit 2
fi

JOB_TMP="$MASTER_ROOT/tmp/${SLURM_JOB_ID}"
mkdir -p "$JOB_TMP" "$PVSG_RUN_ROOT/overfit"
export TMPDIR="$JOB_TMP"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

OVERFIT_ARGUMENTS=(
  --manifest-root "$PVSG_SECTION6_MANIFEST_ROOT"
  --feature-root "$DINO_FEATURE_ROOT"
  --output-root "$PVSG_RUN_ROOT/overfit"
  --run-name "$PVSG_OVERFIT_RUN_NAME"
  --model "${PVSG_OVERFIT_MODEL:-integral}"
  --evolution "${PVSG_OVERFIT_EVOLUTION:-original}"
  --semantic-condition "${PVSG_OVERFIT_SEMANTIC:-hierarchy}"
  --num-examples "${PVSG_OVERFIT_EXAMPLES:-200}"
  --learning-rate "${PVSG_OVERFIT_LEARNING_RATE:-0.001}"
  --max-steps "${PVSG_OVERFIT_MAX_STEPS:-5000}"
  --seed "${PVSG_OVERFIT_SEED:-0}"
  --device cuda
)
if [[ -n "${PVSG_OVERFIT_HIDDEN_DIM:-}" ]]; then
  OVERFIT_ARGUMENTS+=(--hidden-dim "$PVSG_OVERFIT_HIDDEN_DIM")
fi

cd "$TB_REPO_ROOT"
bash "$SCRIPT_DIRECTORY/with_gpu_monitor.sh" \
  uv run --frozen --no-sync python -m experiments.pvsg.overfit \
  "${OVERFIT_ARGUMENTS[@]}"
