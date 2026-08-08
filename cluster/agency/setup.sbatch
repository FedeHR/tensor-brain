#!/usr/bin/env bash
#SBATCH --job-name=agency-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

set -euo pipefail

# Everything the study needs before its first real job, as a batch job: create
# the run roots, sync the Python 3.12 environment, and prove MuJoCo can render.
#
# This exists because interactive access to a compute node is not always
# available. It does both halves of setup on one node, which requires that node
# to have *both* network access and a GPU. If the compute nodes have no network,
# run `cluster/agency/setup.sh` on the login node first -- it is idempotent, so
# this job will then find the environment already synced and only run the render
# check.

REPOSITORY_ROOT="${TB_REPO_ROOT:-${MASTER_ROOT:-/nfs/data8/harjes/MASTER}/tensor-brain}"
source "$REPOSITORY_ROOT/cluster/agency/common.sh"

cd "$TB_REPO_ROOT"
echo "node=$(hostname) code_revision=${AGENCY_CODE_REVISION:-unknown}"

echo "=== 1/2 environment ==="
if ! cluster/agency/setup.sh; then
  echo
  echo "Sync FAILED. The usual cause is a compute node without network access." >&2
  echo "Run 'cluster/agency/setup.sh' on the login node, then resubmit." >&2
  exit 1
fi

echo "=== 2/2 rendering ==="
if ! cluster/agency/render_check.sh; then
  echo
  echo "Rendering FAILED with MUJOCO_GL=$MUJOCO_GL." >&2
  echo "If this node has no usable EGL device, resubmit with MUJOCO_GL=osmesa:" >&2
  echo "  MUJOCO_GL=osmesa cluster/agency/submit_filter.sh setup --partition=<name>" >&2
  exit 1
fi

echo
echo "Setup complete. Next:"
echo "  cluster/agency/submit_filter.sh corpus --partition=<name> --time=01:00:00"
