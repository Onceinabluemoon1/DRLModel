#!/usr/bin/env bash
# Edit the variables below, then run with no args: ./collect_trajs.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VPL_ROOT}"
export PYTHONPATH="${VPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate drl

# --- tyro CLI (CollectTrajsConfig); booleans use --flag / --no-flag ---
#* CollectTrajsConfig:
# Unique folder per run (avoid overwriting): override TS when invoking, e.g.
#   TS="$(date +%Y%m%d_%H%M%S)" ./collect_trajs.sh
#   TS=ablation_seed42 ./collect_trajs.sh
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-v0"
# Per-run DATA_DIR layout:
#   raw_rollouts/*.pkl   raw_rollouts_videos/traj_*.mp4   rollout_dataset.hdf5   rollout_visualization.png   infos.pkl
ENV_NAME="sort-easy-v0"
N=20
OPTS_DISP=--no-disp
OPTS_MULTIMODAL=--multimodal
OPTS_SUCCESS_ONLY=--success-only
# Even goal coverage (0,1,0,1,... per attempt). Omit for random goals via built-in RNG.
OPTS_ALTERNATE_GOAL_MODES=--no-alternate-goal-modes

DATA_DIR="${DATA_ROOT}/${TS}_ntrajs=${N}"
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