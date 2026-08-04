#!/usr/bin/env bash
#SBATCH --job-name=infomec_latengrid
#SBATCH --gres=shard:12
#SBATCH --array=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=scavenger
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/eval_latentgrid_infomec/eval_latentgrid_infomec_%A_%a.out
#
# Slurm array: one latentsweep root per task → scripts/plot_latentgrid_infomec.py
#   • Edit INPUT_DIRS below, then set #SBATCH --array=0-(N-1) for N dirs.
#   • sbatch scripts/submit_plot_latentgrid_infomec.bash
#
# Local (no Slurm): copy-paste block below (works in zsh; conda activate drl first if needed).
: <<'LOCAL_CUDA_RUN'
GPU=4; cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL
INPUT_DIRS=(
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_obs=15_T=1_Q=20000_A=16/VAE/20260728_023552_VAEgrid_pz=T_anneal=F_dec=poslinear_l1=group0.001_encY=T_encS=F"
)
for d in "${INPUT_DIRS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU python scripts/plot_latentgrid_infomec.py \
    --input-dir="$d" \
    --model-selection=final \
    --split=all \
    --device=cuda \
    --dpi=200 \
    --num-bins=10
done
LOCAL_CUDA_RUN

source ~/miniconda3/etc/profile.d/conda.sh
cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL
conda activate drl

#* latentsweep roots (each must contain subruns/ with variant.json + checkpoint)
INPUT_DIRS=(
/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_obs=15_T=25_Q=20000_A=16/VAE/20260804_015609_VAEgrid_pz=F_dec=multi_g0.01_encY=T_encS=F
)

#* plot_latentgrid_infomec CLI (shared across array tasks)
model_selection=final  # best | final
split=all  # all | train | eval
device=cuda  # auto | cuda | cpu
dpi=200
num_bins=10

num_inputs=${#INPUT_DIRS[@]}
last_input_idx=$((num_inputs - 1))

idx="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${idx}" -ge "${num_inputs}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${idx} >= num_inputs=${num_inputs}; skipping."
  exit 0
fi

input_dir="${INPUT_DIRS[$idx]}"
echo "=== [${idx}/${last_input_idx}] plot_latentgrid_infomec: ${input_dir} ==="
python scripts/plot_latentgrid_infomec.py \
  --input-dir="${input_dir}" \
  --model-selection="${model_selection}" \
  --split="${split}" \
  --device="${device}" \
  --dpi="${dpi}" \
  --num-bins="${num_bins}"
