#!/usr/bin/env bash
#SBATCH --job-name=train_qlae
#SBATCH --gres=shard:12
#SBATCH --array=0-6
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/train_qlae/train_qlae_%A_%a.out
#
# sbatch --export=ALL,RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) scripts/submit_train_qlae.bash

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

model_type=QLAE
latent_dim=4
hidden_dim=128
batch_size=256
set_size=16  # -1 = use full S from dataset; else truncate preference sets to this A

loss_aggregation=mean
num_quantized_values=5
quantizer_bound_half_width=1.0  # FSQ R: z_c, z_q in [-R, R]
quantizer_use_ifsq=False  # True → R·(2σ(αz)-1) instead of R·tanh
quantizer_ifsq_alpha=1.6
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
decoder_trunk_layernorm=False

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
# pref_dataset="sort_4goals_obs=15_T=1_Q=20000"
pref_dataset="sort_4goals_obs=15_T=25_Q=20000"

output_dir="/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs"
last_idx=$((${#WEIGHT_DECAYS[@]} - 1))
subrun_pad=${#last_idx}

run_train_qlae() {
  local idx="$1"
  local weight_decay="$2"
  local subrun comment TS_ARGS=()

  subrun=$(printf "%0${subrun_pad}d" "${idx}")
  comment="${subrun}_wd=${weight_decay}"

  if [ -z "${RUN_TIMESTAMP:-}" ] && [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    RUN_TIMESTAMP="job=${SLURM_ARRAY_JOB_ID}"
  fi
  if [ -n "${RUN_TIMESTAMP:-}" ]; then
    TS_ARGS+=(--run_timestamp="${RUN_TIMESTAMP}")
  fi

  python pref_learn/train.py \
    --pref_dataset="${pref_dataset}" \
    --model_type=${model_type} \
    --comment=${comment} \
    --latent_dim=${latent_dim} \
    --hidden_dim=${hidden_dim} \
    --num_quantized_values=${num_quantized_values} \
    --quantizer_bound_half_width=${quantizer_bound_half_width} \
    --quantizer_use_ifsq=${quantizer_use_ifsq} \
    --quantizer_ifsq_alpha=${quantizer_ifsq_alpha} \
    --weight_decay=${weight_decay} \
    --loss_aggregation=${loss_aggregation} \
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
  for idx in "${!WEIGHT_DECAYS[@]}"; do
    echo "=== serial sweep $((idx + 1))/${#WEIGHT_DECAYS[@]}: weight_decay=${WEIGHT_DECAYS[$idx]} ==="
    run_train_qlae "${idx}" "${WEIGHT_DECAYS[$idx]}"
  done
else
  idx="${SLURM_ARRAY_TASK_ID:-0}"
  weight_decay="${WEIGHT_DECAYS[$idx]}"
  run_train_qlae "${idx}" "${weight_decay}"
fi
