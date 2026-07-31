#!/usr/bin/env bash
#SBATCH --job-name=train_vae_grid
#SBATCH --gres=shard:12
#SBATCH --array=0-29%6
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/train_vae_grid/train_vae_grid_%A_%a.out
#
# Grid search over kl_weight (beta) × latent_dim.
# All array tasks share one sweep folder under outputs/<env>/VAE/:
#   <RUN_TIMESTAMP>_latentsweep_learned-prior=..._agg=..._norm-kl=..._anneal=.../
#     subruns/
#       <task>_nz=<L>_beta=.../
#     evals_latentgrid/
#   • Wall-clock (set once at submit on the login node):
#       sbatch --export=ALL,RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) scripts/submit_train_vae_grid.bash
#   • If RUN_TIMESTAMP is unset but this is a Slurm array job, we default to job=<SLURM_ARRAY_JOB_ID>.
#   • All tasks with the same RUN_TIMESTAMP share one W&B group (same tag as sweep folder).
# Each task writes to .../<sweep_folder>/subruns/<array_task_id>_nz=<L>_beta=.../
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
    # 1.0
    # 3.0
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
hidden_dim=128
batch_size=256
set_size=16  # -1 = use full S from dataset; else truncate preference sets to this A

loss_aggregation=mean  # sum | mean
normalize_kl=True
learned_prior=True
# original: logvar = (-8,4) -> std (0.01, 7.3)
# new: logvar = (-6,1) -> std (0.05, 1.64)
prior_logvar_min=-6.0  # learned_prior ARD: clamp prior_logvar
prior_logvar_max=1.0
use_annealing=False
encode_labels=True  # pair encoder: cncat y into pair_mlp (q(z|s1,s2,y))
encode_sources=False  # pair encoder: one-hot GT source_variable after set pool (InfoMEC)

# * === decoder args
decoder_type=poslinear  # poslinear | ellenlinear | slotlinear | odinlinear

# poslinear
decoder_sparsity_weight=0.001  # L1 on theta(s)
decoder_sparsity_mode=group  # elementwise (L1, parts-biased) | group (lasso over heads, axes-biased)
decoder_softplus_beta=2.0  # softplus beta on theta = softplus(w(h))
decoder_trunk_layernorm=True  # LayerNorm on trunk h before w_head

# ellenlinear
# set decoder_type=ellenlinear above; poslinear flags are ignored

# * === end decoder args

# * 2 goals
# pref_dataset="sort_2goals_T=1_Q=5000"
# pref_dataset="sort_2goals_T=10_Q=5000"
# pref_dataset="sort_2goals_T=24_Q=5000"
# pref_dataset="sort_2goals_obs=15_T=1_Q=20000"
# pref_dataset="sort_2goals_obs=15_T=25_Q=20000"
# * 4 goals
# pref_dataset="sort_4goals_T=1_Q=20000"
# pref_dataset="sort_4goals_T=10_Q=20000"
# pref_dataset="sort_4goals_T=25_Q=20000"
pref_dataset="sort_4goals_obs=15_T=1_Q=20000"
# pref_dataset="sort_4goals_obs=15_T=25_Q=20000"


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
  comment="${subrun}_nz=${latent_dim}_beta=${beta}"

  if [ -z "${RUN_TIMESTAMP:-}" ] && [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    RUN_TIMESTAMP="job=${SLURM_ARRAY_JOB_ID}"
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
    --hidden_dim=${hidden_dim} \
    --kl_weight="${beta}" \
    --loss_aggregation=${loss_aggregation} \
    --normalize_kl=${normalize_kl} \
    --learned_prior=${learned_prior} \
    --prior_logvar_min=${prior_logvar_min} \
    --prior_logvar_max=${prior_logvar_max} \
    --use_annealing=${use_annealing} \
    --seed=42 \
    --batch_size=${batch_size} \
    --set_size=${set_size} \
    --decoder_type="${decoder_type}" \
    --decoder_sparsity_weight="${decoder_sparsity_weight}" \
    --decoder_sparsity_mode="${decoder_sparsity_mode}" \
    --decoder_softplus_beta="${decoder_softplus_beta}" \
    --decoder_trunk_layernorm="${decoder_trunk_layernorm}" \
    --encode_labels="${encode_labels}" \
    --encode_sources="${encode_sources}" \
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
    echo "=== serial grid $((idx + 1))/${grid_size}: nz=${latent_dim} beta=${beta} ==="
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
CUDA_VISIBLE_DEVICES=6 RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) bash scripts/submit_train_vae_grid.bash serial
EOF
