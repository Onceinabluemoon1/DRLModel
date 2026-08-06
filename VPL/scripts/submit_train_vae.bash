#!/usr/bin/env bash
#SBATCH --job-name=train_vae
#SBATCH --gres=shard:12
#SBATCH --array=0-10
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/train_vae/train_vae_%A_%a.out
#
# All array tasks share one parent folder under outputs/<env>/<model>/<RUN_TIMESTAMP>_nz=.../:
#   subruns/
#     <task>_beta=.../
#   evals_sweep/
#   • Wall-clock (set once at submit on the login node):
#       sbatch --export=ALL,RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) scripts/submit_train_vae.bash
#   • If RUN_TIMESTAMP is unset but this is a Slurm array job, we default to job=<SLURM_ARRAY_JOB_ID>.
#   • All array tasks with the same RUN_TIMESTAMP share one W&B group (beta sweep).
# Each task writes to .../<RUN_TIMESTAMP>/subruns/<array_task_id>_beta=.../
# variant.json records sweep_param_name=kl_weight, sweep_param_value=<beta>.

source ~/miniconda3/etc/profile.d/conda.sh
cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL
conda activate drl


BETAS=(
    0.001
    0.01
    0.1
    0.3
    1.0
    3.0
    6.0
    10.0
    30.0
    # 100.0
    # 200.0
)
model_type=VAE
latent_dim=4
hidden_dim=128
batch_size=256
set_size=16  # -1 = use full S from dataset; else truncate preference sets to this A

loss_aggregation=mean  # sum | mean
normalize_kl=True
learned_prior=False
# original: logvar = (-8,4) -> std (0.01, 7.3)
# new: logvar = (-6,1) -> std (0.05, 1.64)
prior_logvar_min=-6.0  # learned_prior ARD: clamp prior_logvar
prior_logvar_max=1.0
use_annealing=False
use_nuisance_loss=False  # encoder μ batch auxiliary loss (VAE only)
nuisance_weight=0.01
nuisance_floor=2.0
nuisance_ceil=10.0
encode_labels=True  # pair encoder: concat y into pair_mlp (q(z|s1,s2,y))
encode_sources=False  # pair encoder: one-hot GT source_variable (InfoMEC; relabelled dataset)

# === decoder args
decoder_type=multi  # multi | cond | condsig | condsp | condcat | slot | odin

# head sparsity (training loss; 0 = off)
decoder_sparsity_weight=0.0
# multi only: elementwise (L1 on theta) | group (lasso over heads)
decoder_sparsity_mode=elementwise
# cond | condsig | condsp | condcat: always group L1 on per-head r_j (mode ignored)

# multi
decoder_softplus_beta=1.0  # softplus beta: multi theta / condsp r_j
decoder_trunk_layernorm=True

# cond | condsig | condsp | condcat
decoder_use_rhead_film=False  # SPL-style FiLM head on (z_j, j) instead of concat MLP
# condsig | condcat
decoder_r_max=1.0  # per-head cap in [0, R_max] (condsig / condcat)
# condcat
decoder_n_bins=11  # K bins; index 0 is explicit zero reward
# set decoder_type above; unused flags are ignored by train.py

# ===

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
last_beta_idx=$((${#BETAS[@]} - 1))
subrun_pad=${#last_beta_idx}

run_train_beta() {
  local idx="$1"
  local beta="$2"
  local subrun comment TS_ARGS=()

  subrun=$(printf "%0${subrun_pad}d" "${idx}")
  comment="${subrun}_beta=${beta}"

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
    --comment=${comment} \
    --latent_dim=${latent_dim} \
    --hidden_dim=${hidden_dim} \
    --kl_weight=${beta} \
    --loss_aggregation=${loss_aggregation} \
    --normalize_kl=${normalize_kl} \
    --learned_prior=${learned_prior} \
    --prior_logvar_min=${prior_logvar_min} \
    --prior_logvar_max=${prior_logvar_max} \
    --use_annealing=${use_annealing} \
    --use_nuisance_loss=${use_nuisance_loss} \
    --nuisance_weight=${nuisance_weight} \
    --nuisance_floor=${nuisance_floor} \
    --nuisance_ceil=${nuisance_ceil} \
    --seed=42 \
    --batch_size=${batch_size} \
    --set_size=${set_size} \
    --decoder_type="${decoder_type}" \
    --decoder_sparsity_weight="${decoder_sparsity_weight}" \
    --decoder_sparsity_mode="${decoder_sparsity_mode}" \
    --decoder_softplus_beta="${decoder_softplus_beta}" \
    --decoder_trunk_layernorm="${decoder_trunk_layernorm}" \
    --decoder_use_rhead_film="${decoder_use_rhead_film}" \
    --decoder_r_max="${decoder_r_max}" \
    --decoder_n_bins="${decoder_n_bins}" \
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
  for idx in "${!BETAS[@]}"; do
    echo "=== serial sweep $((idx + 1))/${#BETAS[@]}: beta=${BETAS[$idx]} ==="
    run_train_beta "${idx}" "${BETAS[$idx]}"
  done
else
  idx="${SLURM_ARRAY_TASK_ID:-0}"
  beta="${BETAS[$idx]}"
  run_train_beta "${idx}" "${beta}"
fi


: <<'EOF'
# Slurm array (parallel; one beta per task; shared parent dir via RUN_TIMESTAMP):
sbatch --export=ALL,RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) scripts/submit_train_vae.bash

# Serial loop (one GPU; all betas in order; shared W&B group via RUN_TIMESTAMP):
CUDA_VISIBLE_DEVICES=0 RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) bash scripts/submit_train_vae.bash serial
EOF

# CUDA_VISIBLE_DEVICES=0 python pref_learn/train.py \
#  --env=sort-easy-v0 \
#  --dataset_path=./pref_datasets/sort-easy-v0/relabelled_queries_num5000_q1_s16 \
#  --model_type=VAEClassifier \
#  --comment=sort_beta1_ld8 \
#  --latent_dim=8 \
#  --kl_weight=1.0 \
#  --seed=42 \
#  --batch_size=256 \
#  --n_epochs=500 \
#  --eval_freq=50 \
#  --save_freq=50 \
#  --device=cuda \
#  --logging.output_dir=./logs3 \
#  --logging.online=False