#!/usr/bin/env bash
#SBATCH --job-name=train_tcvae
#SBATCH --gres=shard:12
#SBATCH --array=0-8
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/train_tcvae/train_tcvae_%A_%a.out
#
# All array tasks share one parent folder under outputs/<env>/TCVAE/<RUN_TIMESTAMP>_nz=.../:
#   subruns/
#     <task>_tc=.../
#   evals_sweep/
#   • Wall-clock (set once at submit on the login node):
#       sbatch --export=ALL,RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) scripts/submit_train_tcvae.bash
#   • If RUN_TIMESTAMP is unset but this is a Slurm array job, we default to job=<SLURM_ARRAY_JOB_ID>.
#   • All array tasks with the same RUN_TIMESTAMP share one W&B group (tc_weight sweep).
# Each task writes to .../<RUN_TIMESTAMP>/subruns/<array_task_id>_tc=.../
# variant.json records sweep_param_name=tc_weight, sweep_param_value=<tc_weight>.

source ~/miniconda3/etc/profile.d/conda.sh
cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL
conda activate drl


TC_WEIGHTS=(
    # 0.0000001
    # 0.000001
    # 0.00001
    # way less ^^^
    0.001
    0.01
    0.05
    0.1 # surprisingly good? 
    0.2
    0.3
    1.0
    3.0
    6.0
    # way more vvv
    # 10.0
    # 30.0
    # 100.0
    # 300.0
    # 1000.0
    # 3000.0
)
model_type=TCVAE
latent_dim=32
hidden_dim=128
batch_size=512
set_size=16  # -1 = use full S from dataset; else truncate preference sets to this A

loss_aggregation=mean  # sum | mean
normalize_kl=True
learned_prior=False
use_annealing=False   # ramp tc_weight 0 -> target over n_epochs / annealer_cycles
annealer_cycles=1
annealer_type=cosine
mi_weight=1.0
dwkl_weight=1.0

encode_labels=True  # pair encoder: concat y into pair_mlp (q(z|s1,s2,y))
encode_sources=False  # pair encoder: one-hot GT source_variable (InfoMEC; relabelled dataset)

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
last_tc_idx=$((${#TC_WEIGHTS[@]} - 1))
subrun_pad=${#last_tc_idx}

run_train_tc() {
  local idx="$1"
  local tc_weight="$2"
  local subrun comment TS_ARGS=()

  subrun=$(printf "%0${subrun_pad}d" "${idx}")
  comment="${subrun}_tc=${tc_weight}"

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
    --tc_weight=${tc_weight} \
    --mi_weight=${mi_weight} \
    --dwkl_weight=${dwkl_weight} \
    --loss_aggregation=${loss_aggregation} \
    --normalize_kl=${normalize_kl} \
    --learned_prior=${learned_prior} \
    --use_annealing=${use_annealing} \
    --annealer_cycles=${annealer_cycles} \
    --annealer_type=${annealer_type} \
    --seed=42 \
    --batch_size=${batch_size} \
    --set_size=${set_size} \
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
  for idx in "${!TC_WEIGHTS[@]}"; do
    echo "=== serial sweep $((idx + 1))/${#TC_WEIGHTS[@]}: tc_weight=${TC_WEIGHTS[$idx]} ==="
    run_train_tc "${idx}" "${TC_WEIGHTS[$idx]}"
  done
else
  idx="${SLURM_ARRAY_TASK_ID:-0}"
  tc_weight="${TC_WEIGHTS[$idx]}"
  run_train_tc "${idx}" "${tc_weight}"
fi


: <<'EOF'
# Slurm array (parallel; one tc_weight per task; shared parent dir via RUN_TIMESTAMP):
sbatch --export=ALL,RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) scripts/submit_train_tcvae.bash

# Serial loop (one GPU; all tc_weights in order; shared W&B group via RUN_TIMESTAMP):
CUDA_VISIBLE_DEVICES=0 RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) bash scripts/submit_train_tcvae.bash serial
EOF
