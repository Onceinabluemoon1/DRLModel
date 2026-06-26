#!/usr/bin/env bash
#SBATCH --job-name=plot_latentgrid_heatmap
#SBATCH --gres=shard:12
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/eval_latentgrid_heatmap/eval_latentgrid_heatmap_%A_%a.out
#
# Slurm array: one latentsweep root per task → scripts/plot_latentgrid_heatmap.py
#   • Edit INPUT_DIRS below, then set #SBATCH --array=0-(N-1) for N dirs.
#   • sbatch scripts/submit_plot_latentgrid_heatmap.bash
#
# Local (no Slurm): copy-paste block below (works in zsh; conda activate drl first if needed).
: <<'LOCAL_CUDA_RUN'
GPU=5; cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL; INPUT_DIRS=(
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_T=1_Q=20000_A=16/QLAE/20260625_222934_lgrid_learnpz=F_dec=slotlinear_slotr=sum_encY=T"
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_T=1_Q=20000_A=16/QLAE/20260625_204917_lgrid_learnpz=F_dec=slotlinear_slotr=sum_encY=T"
); for d in "${INPUT_DIRS[@]}"; do CUDA_VISIBLE_DEVICES=$GPU python scripts/plot_latentgrid_heatmap.py --input-dir="$d" --model-selection=final --split=all --device=cuda --dpi=200 --num-bins=10; done
LOCAL_CUDA_RUN

source ~/miniconda3/etc/profile.d/conda.sh
cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL
conda activate drl

#* latentsweep roots (each must contain subruns/ with variant.json + checkpoint)
INPUT_DIRS=(
/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_T=1_Q=20000_A=16/VAE/20260626_012314_lgrid_learnpz=F_anneal=F_dec=slotlinear_slotr=sum_encY=T
/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_T=1_Q=20000_A=16/VAE/20260626_012320_lgrid_learnpz=F_anneal=T_dec=slotlinear_slotr=sum_encY=T
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
