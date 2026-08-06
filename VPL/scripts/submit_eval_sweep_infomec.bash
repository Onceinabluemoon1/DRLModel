#!/usr/bin/env bash
#SBATCH --job-name=plot_infomec_sweep
#SBATCH --gres=shard:12
#SBATCH --array=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/logs_slurm/plot_infomec_sweep/plot_infomec_sweep_%A_%a.out
#
# Slurm array: one sweep root per task → scripts/plot_infomec_sweep.py
#   • Edit SWEEP_DIRS below, then set #SBATCH --array=0-(N-1) for N dirs.
#   • Parallel:
#       sbatch scripts/submit_plot_infomec_sweep.bash
#   • Serial (one GPU, all dirs in order):
#       CUDA_VISIBLE_DEVICES=0 bash scripts/submit_plot_infomec_sweep.bash serial

source ~/miniconda3/etc/profile.d/conda.sh
cd /scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL
conda activate drl

#* sweep roots (each must contain subruns/ with variant.json + checkpoint)
SWEEP_DIRS=(
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_T=25_Q=20000_A=16/QLAE/20260619_214727_ld=4_learned-prior=False_agg=mean_norm-kl=True"
)

#* plot_infomec_sweep CLI (shared across array tasks)
model_selection=final  # best | final
latent_cluster_method=umap  # tsne | umap | none
split=all  # all | train | eval
device=cuda  # auto | cuda | cpu
# Optional overrides (leave empty to use each variant.json default):
batch_size=
set_length=
outdir=

num_sweeps=${#SWEEP_DIRS[@]}
last_sweep_idx=$((num_sweeps - 1))

run_plot_sweep() {
  local idx="$1"
  local sweep_dir="$2"
  local ARGS=(
    scripts/plot_infomec_sweep.py
    --input-dir="${sweep_dir}"
    --model-selection="${model_selection}"
    --latent-cluster-method="${latent_cluster_method}"
    --split="${split}"
    --device="${device}"
  )

  if [ -n "${batch_size}" ]; then
    ARGS+=(--batch-size="${batch_size}")
  fi
  if [ -n "${set_length}" ]; then
    ARGS+=(--set-length="${set_length}")
  fi
  if [ -n "${outdir}" ]; then
    ARGS+=(--outdir="${outdir}")
  fi

  echo "=== [${idx}/${last_sweep_idx}] plot_infomec_sweep: ${sweep_dir} ==="
  python "${ARGS[@]}"
}

if [ "${1:-}" = "serial" ] || [ -n "${SERIAL_SWEEP:-}" ]; then
  for idx in "${!SWEEP_DIRS[@]}"; do
    run_plot_sweep "${idx}" "${SWEEP_DIRS[$idx]}"
  done
else
  idx="${SLURM_ARRAY_TASK_ID:-0}"
  if [ "${idx}" -ge "${num_sweeps}" ]; then
    echo "SLURM_ARRAY_TASK_ID=${idx} >= num_sweeps=${num_sweeps}; skipping."
    exit 0
  fi
  run_plot_sweep "${idx}" "${SWEEP_DIRS[$idx]}"
fi
