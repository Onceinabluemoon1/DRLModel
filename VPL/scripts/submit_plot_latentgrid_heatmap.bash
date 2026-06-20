#!/usr/bin/env bash
#SBATCH --job-name=plot_latentgrid_heatmap
#SBATCH --gres=shard:24
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/plot_latentgrid_heatmap/plot_latentgrid_heatmap_%A_%a.out
#
# Slurm array: one latentsweep root per task → scripts/plot_latentgrid_heatmap.py
#   • Edit INPUT_DIRS below, then set #SBATCH --array=0-(N-1) for N dirs.
#   • sbatch scripts/submit_plot_latentgrid_heatmap.bash

source ~/miniconda3/etc/profile.d/conda.sh
cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL
conda activate drl

#* latentsweep roots (each must contain subruns/ with variant.json + checkpoint)
INPUT_DIRS=(
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_T=25_Q=20000_A=16/QLAE/20260619_224305_latentgrid_learned-prior=False_agg=mean_norm-kl=True"
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_T=25_Q=20000_A=16/TCVAE/20260620_063046_latentgrid_learned-prior=False_agg=mean_norm-kl=True_anneal=True"
)

#* plot_latentgrid_heatmap CLI (shared across array tasks)
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
echo "=== [${idx}/${last_input_idx}] plot_latentgrid_heatmap: ${input_dir} ==="
python scripts/plot_latentgrid_heatmap.py \
  --input-dir="${input_dir}" \
  --model-selection="${model_selection}" \
  --split="${split}" \
  --device="${device}" \
  --dpi="${dpi}" \
  --num-bins="${num_bins}"
