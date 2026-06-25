#!/usr/bin/env bash
#SBATCH --job-name=train_vae_grid
#SBATCH --gres=shard:12
#SBATCH --array=0-47%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/train_vae_grid/train_vae_grid_%A_%a.out
#
# Grid search over kl_weight (beta) × latent_dim.
# All array tasks share one sweep folder under outputs/<env>/VAE/:
#   <RUN_TIMESTAMP>_latentsweep_learned-prior=..._agg=..._norm-kl=..._anneal=.../
#     subruns/
#       <task>_ld=<L>_beta=.../
#     evals_latentgrid/
#   • Wall-clock (set once at submit on the login node):
#       sbatch --export=ALL,RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) scripts/submit_train_vae_grid.bash
#   • If RUN_TIMESTAMP is unset but this is a Slurm array job, we default to arrayjob_<SLURM_ARRAY_JOB_ID>.
#   • All tasks with the same RUN_TIMESTAMP share one W&B group (same tag as sweep folder).
# Each task writes to .../<sweep_folder>/subruns/<array_task_id>_ld=<L>_beta=.../
# variant.json records sweep_param_name=kl_weight, sweep_param_value=<beta> (latent_dim in variant).

source ~/miniconda3/etc/profile.d/conda.sh
cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL
conda activate drl


BETAS=(
    0.0001
    0.001
    0.01
    0.1
    0.3
    1.0
    3.0
    6.0
)
LATENT_DIMS=(
    2
    4
    6
    8
    10
    16
)
model_type=VAE
sweep_folder_tag=latentgrid
learned_prior=False
use_annealing=False
loss_aggregation=mean  # sum | mean
normalize_kl=False

# ── Reward decoder (→ train.py / make_reward_decoder in nets.py) ───────────
decoder_type=linear
decoder_gate_temperature=1.0
decoder_orth_weight=0.0  # linear / odinlinear only; no-op for slotlinear

# pref_dataset="sort_2goals_T=1_Q=5000"
# pref_dataset="sort_2goals_T=10_Q=5000"
# pref_dataset="sort_2goals_T=24_Q=5000"
pref_dataset="sort_4goals_T=1_Q=20000"
# pref_dataset="sort_4goals_T=10_Q=20000"
# pref_dataset="sort_4goals_T=25_Q=20000"

output_dir="/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs"

num_betas=${#BETAS[@]}
num_ld=${#LATENT_DIMS[@]}
grid_size=$((num_betas * num_ld))
last_grid_idx=$((grid_size - 1))
subrun_pad=${#last_grid_idx}

grid_idx_to_ld_beta() {
  local idx="$1"
  local ld_idx beta_idx

  ld_idx=$((idx / num_betas))
  beta_idx=$((idx % num_betas))
  # One line so `read -r latent_dim beta` gets both fields.
  printf '%s %s\n' "${LATENT_DIMS[$ld_idx]}" "${BETAS[$beta_idx]}"
}

run_train_grid() {
  local idx="$1"
  local latent_dim="$2"
  local beta="$3"
  local subrun comment TS_ARGS=()

  subrun=$(printf "%0${subrun_pad}d" "${idx}")
  comment="${subrun}_ld=${latent_dim}_beta=${beta}"

  if [ -z "${RUN_TIMESTAMP:-}" ] && [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    RUN_TIMESTAMP="arrayjob_${SLURM_ARRAY_JOB_ID}"
  fi
  if [ -n "${RUN_TIMESTAMP:-}" ]; then
    TS_ARGS+=(--run_timestamp="${RUN_TIMESTAMP}")
  fi
  # If RUN_TIMESTAMP is still unset, train.py picks datetime per process (non-array local runs).

  python pref_learn/train.py \
    --pref_dataset="${pref_dataset}" \
    --model_type=${model_type} \
    --sweep_folder_tag="${sweep_folder_tag}" \
    --comment=${comment} \
    --latent_dim=${latent_dim} \
    --kl_weight="${beta}" \
    --loss_aggregation=${loss_aggregation} \
    --normalize_kl=${normalize_kl} \
    --learned_prior=${learned_prior} \
    --use_annealing=${use_annealing} \
    --seed=42 \
    --batch_size=256 \
    --decoder_type="${decoder_type}" \
    --decoder_gate_temperature="${decoder_gate_temperature}" \
    --decoder_orth_weight="${decoder_orth_weight}" \
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
    read -r latent_dim beta < <(grid_idx_to_ld_beta "${idx}")
    echo "=== serial grid $((idx + 1))/${grid_size}: ld=${latent_dim} beta=${beta} ==="
    run_train_grid "${idx}" "${latent_dim}" "${beta}"
  done
else
  idx="${SLURM_ARRAY_TASK_ID:-0}"
  if [ "${idx}" -ge "${grid_size}" ]; then
    echo "SLURM_ARRAY_TASK_ID=${idx} >= grid_size=${grid_size}; skipping."
    exit 0
  fi
  read -r latent_dim beta < <(grid_idx_to_ld_beta "${idx}")
  run_train_grid "${idx}" "${latent_dim}" "${beta}"
fi


: <<'EOF'
# Slurm array (parallel; one (ld, beta) pair per task; shared RUN_TIMESTAMP):
# Update #SBATCH --array=0-N to N=(num_betas * num_latent_dims - 1); default 8×6=48 → 0-47.
sbatch --export=ALL,RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) scripts/submit_train_vae_grid.bash

# Serial loop (one GPU; full grid in order; shared W&B group):
CUDA_VISIBLE_DEVICES=0 RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) bash scripts/submit_train_vae_grid.bash serial
EOF
