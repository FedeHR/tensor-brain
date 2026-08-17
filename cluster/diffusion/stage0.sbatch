#!/usr/bin/env bash
#SBATCH --job-name=diffusion-stage0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.harjes@campus.lmu.de

# Stage 0 of the diffusion-decoding assessment, on a full-size model.
#
# The laptop pilot runs a 0.6B MDLM; the models the incumbent paper evaluates are
# LLaDA-8B and Dream-7B, and the whole point of the cluster run is to check that
# the pilot's answer survives scale. Nothing about the measurement changes.
#
#   cluster/diffusion/submit.sh cluster/diffusion/stage0.sbatch
#   DIFFUSION_MODEL=Dream-org/Dream-v0-Instruct-7B \
#     cluster/diffusion/submit.sh cluster/diffusion/stage0.sbatch
#
# An 8B model in bf16 needs ~16 GB of device memory for weights alone; submit with
# SLURM_MEM=48G and a GPU that has at least 24 GB.

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/diffusion/common.sh"

MODEL="${DIFFUSION_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
QUESTIONS="${DIFFUSION_QUESTIONS:-$DIFFUSION_DATA_ROOT/gsm8k/test.jsonl}"
OUTPUT="${DIFFUSION_RUN_ROOT}/$(date +%Y%m%d-%H%M%S)-$(basename "$MODEL")"

echo "model      $MODEL"
echo "questions  $QUESTIONS"
echo "output     $OUTPUT"

cd "$REPOSITORY_ROOT"
mkdir -p "$OUTPUT"

# transformers 5.x removed Qwen3DecoderLayer.attention_type, which the pilot
# model's custom modeling file requires; LLaDA carries its own modeling code and
# is pinned the same way for reproducibility rather than necessity.
uv run --frozen --no-sync --extra coco \
  --with "transformers==${DIFFUSION_TRANSFORMERS:-4.56.2}" \
  --with accelerate --with huggingface_hub \
  python -m experiments.diffusion_heisenberg.run_stage0 \
    --model "$MODEL" \
    --questions "$QUESTIONS" \
    --num-questions "${DIFFUSION_NUM_QUESTIONS:-1000}" \
    --new-tokens "${DIFFUSION_NEW_TOKENS:-64}" \
    --steps "${DIFFUSION_STEPS:-20}" \
    --max-targets "${DIFFUSION_MAX_TARGETS:-16}" \
    --probe-tokens "${DIFFUSION_PROBE_TOKENS:-120}" \
    --min-observations "${DIFFUSION_MIN_OBSERVATIONS:-20}" \
    --model-cache "$DIFFUSION_MODEL_CACHE" \
    --output "$OUTPUT"
