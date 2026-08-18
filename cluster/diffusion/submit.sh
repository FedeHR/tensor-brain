#!/usr/bin/env bash
# Submit a diffusion sbatch script with this cluster's standard resources and logs.
#
# Same wrapper pattern as cluster/pvsg/submit.sh: the resource flags and log
# destinations are identical for every job in the family, and repeating them by
# hand invites the mismatch that silently costs a run. Every value stays
# overridable through the environment.
#
#   cluster/diffusion/submit.sh cluster/diffusion/stage0.sbatch
#   SLURM_MEM=64G SLURM_TIME=6:00:00 cluster/diffusion/submit.sh \
#       cluster/diffusion/stage0.sbatch
#
# SLURM_LOG_ROOT cannot live in an #SBATCH directive, because sbatch parses those
# before any shell variable exists. Resolving it here is why this wrapper exists.

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/diffusion/common.sh"
mkdir -p "$SLURM_LOG_ROOT"

exec sbatch \
  --partition="${SLURM_PARTITION:-all}" \
  --exclude="${SLURM_EXCLUDE:-worker-10}" \
  --gres="${SLURM_GRES:-gpu:1}" \
  --cpus-per-task="${SLURM_CPUS:-4}" \
  --mem="${SLURM_MEM:-48G}" \
  --time="${SLURM_TIME:-6:00:00}" \
  --output="$SLURM_LOG_ROOT/%x-%A.out" \
  --error="$SLURM_LOG_ROOT/%x-%A.err" \
  "$@"
