#!/usr/bin/env bash
#SBATCH --job-name=train_tcvae_grid
#SBATCH --gres=shard:12
#SBATCH --array=0-47%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/train_tcvae_grid/train_tcvae_grid_%A_%a.out
#
# Grid search over tc_weight × latent_dim.
# All array tasks share one sweep folder under outputs/<env>/TCVAE/:
#   <RUN_TIMESTAMP>_latentgrid_learned-prior=..._agg=..._norm-kl=..._anneal=.../
#     subruns/
#       <task>_nz=<L>_tc=.../
#     evals_latentgrid/
#   • Wall-clock (set once at submit on the login node):
#       sbatch --export=ALL,RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) scripts/submit_train_tcvae_grid.bash
#   • If RUN_TIMESTAMP is unset but this is a Slurm array job, we default to job=<SLURM_ARRAY_JOB_ID>.
#   • All tasks with the same RUN_TIMESTAMP share one W&B group (same tag as sweep folder).
# Each task writes to .../<sweep_folder>/subruns/<array_task_id>_nz=<L>_tc=.../
# variant.json records sweep_param_name=tc_weight, sweep_param_value=<tc> (latent_dim in variant).

source ~/miniconda3/etc/profile.d/conda.sh
cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL
conda activate drl


TC_WEIGHTS=(
    0.0001
    0.001
    0.01
    0.1
    1.0
    3.0
    6.0
)
LATENT_DIMS=(
    2
    4
    8
    10
    16
    24
)
model_type=TCVAE
sweep_folder_tag=latentgrid
hidden_dim=128
batch_size=512
set_size=16  # -1 = use full S from dataset; else truncate preference sets to this A

loss_aggregation=mean  # sum | mean
normalize_kl=True
learned_prior=False
prior_logvar_min=-8.0  # learned_prior ARD: clamp prior_logvar
prior_logvar_max=4.0
use_annealing=False   # ramp tc_weight 0 -> target over n_epochs / annealer_cycles
annealer_cycles=1
annealer_type=cosine
mi_weight=1.0
dwkl_weight=1.0
encode_labels=True  # pair encoder: concat y into pair_mlp (q(z|s1,s2,y))
encode_sources=False  # pair encoder: one-hot GT source_variable (InfoMEC; relabelled dataset)

# === decoder args
decoder_type=poslinear  # poslinear | ellenlinear | slotlinear | odinlinear

# poslinear
decoder_sparsity_weight=0.0  # L1 on theta(s)
decoder_sparsity_mode=elementwise  # elementwise (L1, parts-biased) | group (lasso over heads, axes-biased)
decoder_softplus_beta=1.0  # softplus beta on theta = softplus(w(h))
decoder_trunk_layernorm=False

# ellenlinear
# set decoder_type=ellenlinear above; poslinear flags are ignored

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

num_tc=${#TC_WEIGHTS[@]}
num_ld=${#LATENT_DIMS[@]}
grid_size=$((num_tc * num_ld))
last_grid_idx=$((grid_size - 1))
subrun_pad=${#last_grid_idx}

grid_idx_to_ld_tc() {
  local idx="$1"
  local ld_idx tc_idx

  ld_idx=$((idx / num_tc))
  tc_idx=$((idx % num_tc))
  # One line so `read -r latent_dim tc_weight` gets both fields.
  printf '%s %s\n' "${LATENT_DIMS[$ld_idx]}" "${TC_WEIGHTS[$tc_idx]}"
}

run_train_grid() {
  local idx="$1"
  local latent_dim="$2"
  local tc_weight="$3"
  local subrun comment TS_ARGS=()

  subrun=$(printf "%0${subrun_pad}d" "${idx}")
  comment="${subrun}_nz=${latent_dim}_tc=${tc_weight}"

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
    --tc_weight=${tc_weight} \
    --mi_weight=${mi_weight} \
    --dwkl_weight=${dwkl_weight} \
    --loss_aggregation=${loss_aggregation} \
    --normalize_kl=${normalize_kl} \
    --learned_prior=${learned_prior} \
    --prior_logvar_min=${prior_logvar_min} \
    --prior_logvar_max=${prior_logvar_max} \
    --use_annealing=${use_annealing} \
    --annealer_cycles=${annealer_cycles} \
    --annealer_type=${annealer_type} \
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
    read -r latent_dim tc_weight < <(grid_idx_to_ld_tc "${idx}")
    echo "=== serial grid $((idx + 1))/${grid_size}: nz=${latent_dim} tc=${tc_weight} ==="
    run_train_grid "${idx}" "${latent_dim}" "${tc_weight}"
  done
else
  idx="${SLURM_ARRAY_TASK_ID:-0}"
  if [ "${idx}" -ge "${grid_size}" ]; then
    echo "SLURM_ARRAY_TASK_ID=${idx} >= grid_size=${grid_size}; skipping."
    exit 0
  fi
  read -r latent_dim tc_weight < <(grid_idx_to_ld_tc "${idx}")
  run_train_grid "${idx}" "${latent_dim}" "${tc_weight}"
fi


: <<'EOF'
# Slurm array (parallel; one (ld, tc_weight) pair per task; shared RUN_TIMESTAMP):
# Update #SBATCH --array=0-N to N=(num_tc_weights * num_latent_dims - 1); default 8×6=48 → 0-47.
sbatch --export=ALL,RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) scripts/submit_train_tcvae_grid.bash

# Serial loop (one GPU; full grid in order; shared W&B group):
CUDA_VISIBLE_DEVICES=0 RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) bash scripts/submit_train_tcvae_grid.bash serial
EOF
