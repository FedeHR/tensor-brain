#!/usr/bin/env bash
#SBATCH --job-name=pvsg-ambiguity-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f.harjes@campus.lmu.de

# Does PVSG contain observations that memory could help with?
#
# This is the go/no-go for the streaming and episodic experiments, and it runs
# before either is built. Under the blocked protocol every track is enrolled from
# the observation window and every evaluation observation carries the delay since
# its last sighting, so re-identification from appearance alone is a nearest-track
# retrieval over cached features. No training, no gradients, no GPU.
#
# Read the report in this order.
#
#   rules.centroid.top1        how often frozen DINO already answers correctly.
#                              1 - top1 bounds what any memory mechanism could add.
#   rules.centroid.ambiguous_at
#                              the share of observations decided by less than each
#                              margin. This is the population where temporal
#                              context could decide the answer.
#   by_delay                   whether that population grows with time since the
#                              last sighting, which is what would make it a
#                              *memory* problem rather than a hard-crop problem.
#   by_mask_area               the honest caveat. An ambiguous population sitting
#                              entirely in the smallest masks is probably
#                              irreducible rather than recoverable.
#   rules.nearest_view         the same retrieval against the single best view
#                              instead of the accumulated prototype. If the
#                              centroid does not beat it, averaging across
#                              occurrences buys nothing on this data, which is
#                              the premise the index-feedback argument rests on.
#
# The decision. A small-margin population of a few percent means no memory
# mechanism will move a headline number on PVSG, and the streaming experiment
# should not be built -- an effect that size sits under the 3.8 pp between-seed
# spread the pair campaign already measured. A fifth of the data is a target
# worth building for.
#
# This is CPU and I/O bound: it loads one cached feature artifact per video and
# does a few small matrix products. Submit it without a GPU.
#
#   SLURM_GRES=none SLURM_CPUS=4 SLURM_MEM=16G SLURM_TIME=2:00:00 \
#     cluster/pvsg/submit.sh cluster/pvsg/ambiguity_audit.sbatch

set -euo pipefail

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/pvsg/common.sh"

OUTPUT_ROOT="$PVSG_RUN_ROOT/${AUDIT_RUN_GROUP:-ambiguity-audit}"

export TMPDIR="$MASTER_ROOT/tmp/${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR" "$OUTPUT_ROOT"

cd "$TB_REPO_ROOT"
uv run --frozen --no-sync python -m experiments.pvsg.ambiguity_audit \
  --manifest-root "$PVSG_SECTION6_MANIFEST_ROOT" \
  --feature-root "$DINO_FEATURE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --enrollment-manifest "${AUDIT_ENROLLMENT_MANIFEST:-blocked/train_objects.jsonl}" \
  --evaluation-manifest "${AUDIT_EVALUATION_MANIFEST:-blocked/evaluation_objects.jsonl}" \
  --write-observations
