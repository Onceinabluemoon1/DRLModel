#!/usr/bin/env bash
#SBATCH --job-name=train_vae
#SBATCH --gres=shard:12
#SBATCH --array=0-10
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/train_vae/train_vae_%A_%a.out
#
# All array tasks share one parent folder under logs3/<env>/<model>/<RUN_TIMESTAMP>/:
#   • Wall-clock (set once at submit on the login node):
#       sbatch --export=ALL,RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) scripts/submit_train_vae.bash
#   • If RUN_TIMESTAMP is unset but this is a Slurm array job, we default to arrayjob_<SLURM_ARRAY_JOB_ID>.
#   • All array tasks with the same RUN_TIMESTAMP share one W&B group (beta sweep).
# Each task writes to .../<RUN_TIMESTAMP>/<array_task_id>_beta=.../
# variant.json records sweep_param_name=kl_weight, sweep_param_value=<beta>.

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
    10.0
    30.0
    # 100.0
    # 200.0
)
model_type=VAE
# model_type=TCVAE
latent_dim=4
learned_prior=True
use_annealing=False
loss_aggregation=mean  # sum | mean

# pref_dataset="sort_2goals_T=1"
# pref_dataset="sort_4goals_T=1"
pref_dataset="sort_4goals_T=1_Q=20000"
# pref_dataset="sort_4goals_T=10"

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
    RUN_TIMESTAMP="arrayjob_${SLURM_ARRAY_JOB_ID}"
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
    --kl_weight=${beta} \
    --loss_aggregation=${loss_aggregation} \
    --learned_prior=${learned_prior} \
    --use_annealing=${use_annealing} \
    --seed=42 \
    --batch_size=256 \
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