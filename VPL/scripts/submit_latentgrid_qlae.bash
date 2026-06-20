#!/usr/bin/env bash
#SBATCH --job-name=train_qlae_grid
#SBATCH --gres=shard:12
#SBATCH --array=0-41%7
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/train_qlae_grid/train_qlae_grid_%A_%a.out
#
# Grid search over weight_decay × latent_dim.
# All array tasks share one sweep folder under outputs/<env>/QLAE/:
#   <RUN_TIMESTAMP>_latentgrid_learned-prior=..._agg=..._norm-kl=.../
#     subruns/
#       <task>_ld=<L>_wd=.../
#     evals_latentgrid/
#   • Wall-clock (set once at submit on the login node):
#       sbatch --export=ALL,RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) scripts/submit_latentgrid_qlae.bash
#   • If RUN_TIMESTAMP is unset but this is a Slurm array job, we default to arrayjob_<SLURM_ARRAY_JOB_ID>.
#   • All tasks with the same RUN_TIMESTAMP share one W&B group (same tag as sweep folder).
# Each task writes to .../<sweep_folder>/subruns/<array_task_id>_ld=<L>_wd=.../
# variant.json records sweep_param_name=weight_decay, sweep_param_value=<wd> (latent_dim in variant).

source ~/miniconda3/etc/profile.d/conda.sh
cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL
conda activate drl

WEIGHT_DECAYS=(
    0.0
    0.0001
    0.001
    0.01
    0.1
    1.0
    3.0
)
LATENT_DIMS=(
    4
    8
    16
    32
    64
    128
)

model_type=QLAE
sweep_folder_tag=latentgrid
hidden_dim=128
batch_size=256
num_quantized_values=6
loss_aggregation=mean
set_size=16  # -1 = use full S from dataset; else truncate preference sets to this A

# pref_dataset="sort_2goals_T=1_Q=5000"
# pref_dataset="sort_2goals_T=10_Q=5000"
# pref_dataset="sort_2goals_T=24_Q=5000"
# pref_dataset="sort_4goals_T=1_Q=20000"
# pref_dataset="sort_4goals_T=10_Q=20000"
pref_dataset="sort_4goals_T=25_Q=20000"

output_dir="/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs"

num_wd=${#WEIGHT_DECAYS[@]}
num_ld=${#LATENT_DIMS[@]}
grid_size=$((num_wd * num_ld))
last_grid_idx=$((grid_size - 1))
subrun_pad=${#last_grid_idx}

grid_idx_to_ld_wd() {
  local idx="$1"
  local ld_idx wd_idx

  ld_idx=$((idx / num_wd))
  wd_idx=$((idx % num_wd))
  printf '%s %s\n' "${LATENT_DIMS[$ld_idx]}" "${WEIGHT_DECAYS[$wd_idx]}"
}

run_train_grid() {
  local idx="$1"
  local latent_dim="$2"
  local weight_decay="$3"
  local subrun comment TS_ARGS=()

  subrun=$(printf "%0${subrun_pad}d" "${idx}")
  comment="${subrun}_ld=${latent_dim}_wd=${weight_decay}"

  if [ -z "${RUN_TIMESTAMP:-}" ] && [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    RUN_TIMESTAMP="arrayjob_${SLURM_ARRAY_JOB_ID}"
  fi
  if [ -n "${RUN_TIMESTAMP:-}" ]; then
    TS_ARGS+=(--run_timestamp="${RUN_TIMESTAMP}")
  fi

  python pref_learn/train.py \
    --pref_dataset="${pref_dataset}" \
    --model_type=${model_type} \
    --sweep_folder_tag="${sweep_folder_tag}" \
    --comment=${comment} \
    --latent_dim=${latent_dim} \
    --hidden_dim=${hidden_dim} \
    --num_quantized_values=${num_quantized_values} \
    --weight_decay=${weight_decay} \
    --loss_aggregation=${loss_aggregation} \
    --seed=42 \
    --batch_size=${batch_size} \
    --set_size=${set_size} \
    --n_epochs=500 \
    --eval_freq=50 \
    --save_freq=50 \
    --device=cuda \
    --logging.output_dir=${output_dir} \
    --logging.online=True \
    "${TS_ARGS[@]}"
}

if [ "${1:-}" = "serial" ] || [ -n "${SERIAL_SWEEP:-}" ]; then
  RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
  export RUN_TIMESTAMP
  for idx in $(seq 0 $((grid_size - 1))); do
    read -r latent_dim weight_decay < <(grid_idx_to_ld_wd "${idx}")
    echo "=== serial grid $((idx + 1))/${grid_size}: ld=${latent_dim} wd=${weight_decay} ==="
    run_train_grid "${idx}" "${latent_dim}" "${weight_decay}"
  done
else
  idx="${SLURM_ARRAY_TASK_ID:-0}"
  if [ "${idx}" -ge "${grid_size}" ]; then
    echo "SLURM_ARRAY_TASK_ID=${idx} >= grid_size=${grid_size}; skipping."
    exit 0
  fi
  read -r latent_dim weight_decay < <(grid_idx_to_ld_wd "${idx}")
  run_train_grid "${idx}" "${latent_dim}" "${weight_decay}"
fi

: <<'EOF'
# Slurm array (parallel; one (ld, wd) pair per task; shared RUN_TIMESTAMP):
# Update #SBATCH --array=0-N to N=(num_weight_decays * num_latent_dims - 1); default 7×6=42 → 0-41.
sbatch --export=ALL,RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) scripts/submit_latentgrid_qlae.bash

# Serial loop (one GPU; full grid in order; shared W&B group):
CUDA_VISIBLE_DEVICES=0 RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) bash scripts/submit_latentgrid_qlae.bash serial
EOF
