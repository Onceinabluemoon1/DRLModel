#!/usr/bin/env bash
#SBATCH --job-name=plot_latentgrid_rewardmap
#SBATCH --gres=shard:24
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/eval_latentgrid_rewardmap/eval_latentgrid_rewardmap_%A_%a.out
#
# Slurm array: one latentsweep root per task → scripts/plot_latentgrid_rewardmap.py
#
# Local (no Slurm): copy-paste block below (conda activate drl first if needed).
: <<'LOCAL_CUDA_RUN'
GPU=4; cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL
INPUT_DIRS=(
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_obs=15_T=1_Q=20000_A=16/VAE/20260727_144208_VAEgrid_pz=F_anneal=F_dec=poslinear_sparw=0.001_sparm=group_encY=T_encS=F"
)
for d in "${INPUT_DIRS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU python scripts/plot_latentgrid_rewardmap.py \
    --input-dir="$d" \
    --recon-top-k=4 \
    --recon-loss-threshold=0.1 \
    --plot-xy=ee \
    --reward-mode=btl \
    --btl-unit=traj \
    --btl-traj-scatter=random \
    --btl-logit-scaling=train \
    --viz-data-frac=0.6 \
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
"/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_obs=15_T=1_Q=20000_A=16/VAE/20260726_043422_VAEgrid_pz=F_anneal=F_dec=poslinear_sparw=0.001_sparm=group_encY=T_encS=F"
)

#* sweep HP auto-selected from latest evals_latentgrid/ metrics:
#*   rows: any cell with recon_loss <= threshold; top-k lowest mean recon over passing cells
#*   tie-break: max InfoM over passing cells (not whole-row max)
#* override: sweep_param_value=0.01  (pass --sweep-param-value)

#* outputs (sort-easy): evals_rewardmap/<ts>_…/btl_rewards/nz=<nz>_recon=…_M=…_E=…_C=…/
#*   nz=<nz>_00left.png … nz=<nz>_07bottom_right.png, nz=<nz>_08aggregate.png

#* plot_latentgrid_rewardmap CLI
topk=4  # among HP rows with ≥1 passing cell, keep up to this many lowest mean passing recon
recon_loss_threshold=0.1  # row filter (any cell) + top-k/mean recon + InfoM tie-break on passing cells
reward_mode=btl  # raw | btl | both
model_selection=final  # best | final
split=all  # all | train | eval
device=cuda  # auto | cuda | cpu
dpi=150
prob_max_comp=512
prob_chunk_size=64
prob_comp_chunk_size=64
viz_data_frac=0.6  # (0, 1]; e.g. 0.1 for faster debug scatter
plot_xy=ee  # ee | obj
btl_unit=traj  # frame | traj
btl_traj_scatter=random  # last | random (frame pick for xy / frame-BTL s_A)
#* btl_logit_scaling: BTL logit divisor before sigmoid (per-head maps)
#*   train    → model.scaling (= training reward_scaling)
#*   per_head → train / nz            if btl_unit=traj
#*              train / (T · nz)      if btl_unit=frame
#*   (training BTL uses summed heads; per_head downscales single-head logits)
btl_logit_scaling=train  # train | per_head
#* sort-easy: always stratify by GT pref mode (no CLI flag)

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
  --prob-comp-chunk-size="${prob_comp_chunk_size}" \
  --viz-data-frac="${viz_data_frac}" \
  --plot-xy="${plot_xy}" \
  --btl-unit="${btl_unit}" \
  --btl-traj-scatter="${btl_traj_scatter}" \
  --btl-logit-scaling="${btl_logit_scaling}"
