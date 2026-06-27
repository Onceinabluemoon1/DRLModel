#!/usr/bin/env bash
set -euo pipefail

#* --- edit these ---
env="sort-easy-2goals-v0"
# env="sort-easy-4goals-v0"
n_trajs=5000
max_horizon=25
plot_random_trajs=10

seed=-1
n_queries=20000 # Q
set_size=16 # A
segment_len=1 # T

run_pref_dataset=1 # 0 = rollouts + HDF5 only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${VPL_ROOT}/.." && pwd)"
cd "${VPL_ROOT}"
export PYTHONPATH="${VPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate drl

ts="$(date +%Y%m%d_%H%M%S)"
data_dir="${REPO_ROOT}/data/ravens/${env}/${ts}_ntrajs=${n_trajs}"
h5_fp="${data_dir}/rollout_dataset.hdf5"

echo "data_dir=${data_dir}"
echo "h5_fp=${h5_fp}"

# * collect oracle rollouts (15-dim obs)
python "${VPL_ROOT}/utils/collect_trajs.py" \
  --data-dir="${data_dir}" \
  --no-disp \
  --env-name="${env}" \
  --n="${n_trajs}" \
  --multimodal \
  --success-only \
  --no-alternate-goal-modes \
  --plot-random-trajs="${plot_random_trajs}"

# * pkl → HDF5
python "${VPL_ROOT}/utils/demos2dataset.py" \
  --rollouts-dir="${data_dir}/raw_rollouts" \
  --output-h5="${h5_fp}" \
  --max-horizon="${max_horizon}"

if [[ "${run_pref_dataset}" == "1" ]]; then
  python -m pref_learn.create_dataset_infomec \
    --env="${env}" \
    --num_query="${n_queries}" \
    --segment_len="${segment_len}" \
    --set_size="${set_size}" \
    --offline_dataset_h5="${h5_fp}" \
    --store_source_mode=True \
    --data_seed="${seed}"
fi
