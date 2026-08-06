#!/usr/bin/env bash
# Submit a PVSG sbatch script with this cluster's standard resources and log paths.
#
# The resource flags and the log destinations are identical for every PVSG job, and
# repeating them by hand invites the kind of mismatch that silently costs a run. Each
# value stays overridable through the environment.
#
#   cluster/pvsg/submit.sh --array=0-1 cluster/pvsg/pair_learned_gate.sbatch
#   PAIR_SEED=1 PAIR_PROTOCOL=blocked cluster/pvsg/submit.sh --array=3-4 \
#       cluster/pvsg/pair_known_entities.sbatch
#
# SLURM_LOG_ROOT cannot live in an #SBATCH directive, because sbatch parses those before
# any shell variable exists. Resolving it here is why this wrapper exists at all.

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/pvsg/common.sh"
mkdir -p "$SLURM_LOG_ROOT"

exec sbatch \
  --partition="${SLURM_PARTITION:-all}" \
  --exclude="${SLURM_EXCLUDE:-worker-10}" \
  --gres="${SLURM_GRES:-gpu:1}" \
  --cpus-per-task="${SLURM_CPUS:-3}" \
  --mem="${SLURM_MEM:-16G}" \
  --time="${SLURM_TIME:-3:00:00}" \
  --output="$SLURM_LOG_ROOT/%x-%A_%a.out" \
  --error="$SLURM_LOG_ROOT/%x-%A_%a.err" \
  "$@"
