#!/usr/bin/env bash
#SBATCH --job-name=plot_latentgrid_rewardmap
#SBATCH --gres=shard:24
#SBATCH --array=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/eval_latentgrid_rewardmap/eval_latentgrid_rewardmap_%A_%a.out
#
# Slurm array: one latentsweep root per task → scripts/plot_latentgrid_rewardmap.py
#
# Local (no Slurm): copy-paste block below (conda activate drl first if needed).
: <<'LOCAL_CUDA_RUN'
GPU=6; cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL
INPUT_DIRS=(
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_obs=15_T=1_Q=20000_A=16/VAE/20260720_154510_VAEgrid_pz=F_anneal=F_dec=poslinear_sparw=0.001_sparm=group_encY=T_encS=T"
)
for d in "${INPUT_DIRS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU python scripts/plot_latentgrid_rewardmap.py \
    --input-dir="$d" \
    --recon-top-k=4 \
    --recon-loss-threshold=0.1 \
    --reward-mode=prob \
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
/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_obs=15_T=1_Q=20000_A=16/VAE/20260720_233328_VAEgrid_pz=F_anneal=F_dec=poslinear_sparw=0.001_sparm=group_encY=T_encS=T
)

#* sweep HP auto-selected from latest evals_latentgrid/ metrics:
#*   optional recon_loss_threshold → up to topk lowest avg recon_loss HPs → max InfoM
#* override: sweep_param_value=0.01  (pass --sweep-param-value)

#* outputs: evals_rewardmap/<timestamp>_<beta|wd|tc>=<value>/prob_rewards/ ...

#* plot_latentgrid_rewardmap CLI
topk=4  # among threshold-passing HPs, keep up to this many lowest avg recon_loss
recon_loss_threshold=0.1  # uncomment: only HPs with avg recon_loss <= threshold
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
recon_loss_threshold_args=()
if [ -n "${recon_loss_threshold:-}" ]; then
  recon_loss_threshold_args=(--recon-loss-threshold="${recon_loss_threshold}")
fi

echo "=== [${idx}/${last_input_idx}] plot_latentgrid_rewardmap: ${input_dir} (auto sweep HP, topk=${topk}${recon_loss_threshold:+, recon_loss_threshold=${recon_loss_threshold}}) ==="
python scripts/plot_latentgrid_rewardmap.py \
  --input-dir="${input_dir}" \
  --recon-top-k="${topk}" \
  "${recon_loss_threshold_args[@]}" \
  --reward-mode="${reward_mode}" \
  --model-selection="${model_selection}" \
  --split="${split}" \
  --device="${device}" \
  --dpi="${dpi}" \
  --prob-max-comp="${prob_max_comp}" \
  --prob-chunk-size="${prob_chunk_size}" \
  --prob-comp-chunk-size="${prob_comp_chunk_size}"
