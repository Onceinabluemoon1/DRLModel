#!/usr/bin/env bash
# Edit the variables below, then run with no args: ./collect_trajs.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VPL_ROOT}"
export PYTHONPATH="${VPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate drl

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="$(cd "${VPL_ROOT}/.." && pwd)"

# --- tyro CLI (CollectTrajsConfig); booleans use --flag / --no-flag ---
#* CollectTrajsConfig:
# Unique folder per run (avoid overwriting): override TS when invoking, e.g.
#   TS="$(date +%Y%m%d_%H%M%S)" ./collect_trajs.sh
#   TS=ablation_seed42 ./collect_trajs.sh
# Switch task / output bucket (defaults ``ENV_NAME=sort-easy-v0``):
#   ENV_NAME=sort-easy-4goals-v0 ./collect_trajs.sh
# Gym env id (must match registered ``jaxrl_m`` env); output goes under ``data/ravens/<ENV_NAME>/``.
ENV_NAME="sort-easy-4goals-v0"
N=5000
OPTS_DISP=--no-disp
OPTS_MULTIMODAL=--multimodal
OPTS_SUCCESS_ONLY=--success-only
# Even goal coverage (0,1,0,1,... per attempt). Omit for random goals via built-in RNG.
OPTS_ALTERNATE_GOAL_MODES=--no-alternate-goal-modes

DATA_DIR="${REPO_ROOT}/data/ravens/${ENV_NAME}/${TS}_ntrajs=${N}"
echo "collect_trajs: DATA_DIR=${DATA_DIR}"

#* DemosToH5Config:
HDF5_OUT="${DATA_DIR}/rollout_dataset.hdf5"
MAX_HORIZON=25
echo "demos2dataset: rollouts=${DATA_DIR}/raw_rollouts HDF5_OUT=${HDF5_OUT}"

python "${SCRIPT_DIR}/collect_trajs.py" \
  --data-dir="${DATA_DIR}" \
  "${OPTS_DISP}" \
  --env-name="${ENV_NAME}" \
  --n="${N}" \
  "${OPTS_MULTIMODAL}" \
  "${OPTS_SUCCESS_ONLY}" \
  "${OPTS_ALTERNATE_GOAL_MODES}"

python "${SCRIPT_DIR}/demos2dataset.py" \
  --rollouts-dir="${DATA_DIR}/raw_rollouts" \
  --output-h5="${HDF5_OUT}" \
  --max-horizon="${MAX_HORIZON}"