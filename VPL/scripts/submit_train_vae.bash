#!/usr/bin/env bash
#SBATCH --job-name=train_vae
#SBATCH --gres=shard:24
#SBATCH --array=0-10
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/train_vae/train_vae_%A_%a.out
#
# All array tasks share one parent folder under logs3/<env>/<model>/<RUN_TIMESTAMP>/:
#   • Wall-clock (set once at submit on the login node):
#       sbatch --export=ALL,RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) scripts/submit_train_vae.bash
#   • If RUN_TIMESTAMP is unset but this is a Slurm array job, we default to arrayjob_<SLURM_ARRAY_JOB_ID>.
# Each task writes to .../<RUN_TIMESTAMP>/<array_task_id>_beta=..._ld=.../

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
    100.0
    200.0
)
beta=${BETAS[${SLURM_ARRAY_TASK_ID}]}
latent_dim=8

env=sort-easy-v0
dataset_path=./pref_datasets/${env}/relabelled_queries_num5000_q1_s16
subrun="${SLURM_ARRAY_TASK_ID:-0}"
comment="${subrun}_beta=${beta}_ld=${latent_dim}"
# comment="beta=${beta}_ld=${latent_dim}"

TS_ARGS=()
if [ -z "${RUN_TIMESTAMP:-}" ] && [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
  RUN_TIMESTAMP="arrayjob_${SLURM_ARRAY_JOB_ID}"
fi
if [ -n "${RUN_TIMESTAMP:-}" ]; then
  TS_ARGS+=(--run_timestamp="${RUN_TIMESTAMP}")
fi
# If RUN_TIMESTAMP is still unset, train.py picks datetime per process (non-array local runs).

python pref_learn/train.py \
  --env=${env} \
  --dataset_path=${dataset_path} \
  --model_type=VAEClassifier \
  --comment=${comment} \
  --latent_dim=${latent_dim} \
  --kl_weight=${beta} \
  --seed=42 \
  --batch_size=256 \
  --n_epochs=500 \
  --eval_freq=50 \
  --save_freq=50 \
  --device=cuda \
  --logging.output_dir=./logs3 \
  --logging.online=False \
  "${TS_ARGS[@]}"

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