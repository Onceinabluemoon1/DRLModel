#!/usr/bin/env bash
#SBATCH --job-name=plot_latentgrid_rewardmap
#SBATCH --gres=shard:24
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/eval_latentgrid_rewardmap/eval_latentgrid_rewardmap_%A_%a.out
#
# Slurm array: one latentsweep root per task → scripts/plot_latentgrid_rewardmap.py
#
# Local (no Slurm): copy-paste block below (conda activate drl first if needed).
: <<'LOCAL_CUDA_RUN'
GPU=5; cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL
topk=4
INPUT_DIRS=(
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_obs=15_T=25_Q=20000_A=16/VAE/20260627_053414_lgrid_learnpz=F_anneal=F_dec=slotlinear_slotr=sum_encY=T"
)
for d in "${INPUT_DIRS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU python scripts/plot_latentgrid_rewardmap.py \
    --input-dir="$d" \
    --recon-top-k="${topk}" \
    --reward-mode=both \
    --model-selection=final \
    --split=all \
    --device=cuda \
    --dpi=150
done
LOCAL_CUDA_RUN

source ~/miniconda3/etc/profile.d/conda.sh
cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL
conda activate drl

#* latentsweep roots (sum-over-Z decoders only: linear / odinlinear / slotlinear / ellenlinear / offsetlinear)
INPUT_DIRS=(
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_obs=15_T=1_Q=20000_A=16/VAE/20260627_230120_lgrid_learnpz=F_anneal=F_dec=offsetlinear_encY=T"
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_obs=15_T=1_Q=20000_A=16/QLAE/20260629_003701_lgrid_nv=6_dec=offsetlinear_encY=T"
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_obs=15_T=25_Q=20000_A=16/VAE/20260627_230145_lgrid_learnpz=F_anneal=F_dec=offsetlinear_encY=T"
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_obs=15_T=25_Q=20000_A=16/QLAE/20260627_230147_lgrid_nv=6_dec=offsetlinear_encY=T"
)

#* sweep HP auto-selected from latest evals_latentgrid/ metrics:
#*   lowest topk avg train/recon_loss rows → highest max InfoM among those
#* override: sweep_param_value=0.01  (pass --sweep-param-value)

#* plot_latentgrid_rewardmap CLI
topk=4  # lowest avg recon_loss rows kept before max-InfoM tie-break
reward_mode=prob  # raw | prob | both
model_selection=final  # best | final
split=all  # all | train | eval
device=cuda  # auto | cuda | cpu
dpi=150
prob_max_comp=512
prob_chunk_size=64
prob_comp_chunk_size=64

num_inputs=${#INPUT_DIRS[@]}
last_input_idx=$((num_inputs - 1))

idx="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${idx}" -ge "${num_inputs}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${idx} >= num_inputs=${num_inputs}; skipping."
  exit 0
fi

input_dir="${INPUT_DIRS[$idx]}"
echo "=== [${idx}/${last_input_idx}] plot_latentgrid_rewardmap: ${input_dir} (auto sweep HP, topk=${topk}) ==="
python scripts/plot_latentgrid_rewardmap.py \
  --input-dir="${input_dir}" \
  --recon-top-k="${topk}" \
  --reward-mode="${reward_mode}" \
  --model-selection="${model_selection}" \
  --split="${split}" \
  --device="${device}" \
  --dpi="${dpi}" \
  --prob-max-comp="${prob_max_comp}" \
  --prob-chunk-size="${prob_chunk_size}" \
  --prob-comp-chunk-size="${prob_comp_chunk_size}"
