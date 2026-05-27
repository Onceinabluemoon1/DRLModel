# # #* eval single run
# python scripts/plot_infomec.py \
#   --input_dir "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_T=1/VAE/20260523_164553_ld=4_learned-prior=True/06_beta=3.0" \
#   --latent-cluster-method umap \
#   --model-selection final


# # #* eval sweep:
# python scripts/plot_infomec_sweep.py \
#   --input_dir "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_T=10/VAE/20260518_154659_ld=4_learned-prior=False" \
#   --latent-cluster-method umap \
#   --model-selection final


#* beta sweeps: UMAP + final checkpoint (latest checkpoints/model_<epoch>.pt per subrun)
SWEEP_DIRS=(
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_2goals_T=1/VAE/20260525_152126_ld=4_learned-prior=True"
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_T=1/VAE/20260523_164553_ld=4_learned-prior=True"
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_T=1/VAE/20260525_155919_ld=8_learned-prior=True"
  "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_T=1_Q=20000/VAE/20260525_163605_ld=4_learned-prior=True"
)
for sweep_dir in "${SWEEP_DIRS[@]}"; do
  echo "=== sweep eval: ${sweep_dir} ==="
  python scripts/plot_infomec_sweep.py \
    --input_dir "${sweep_dir}" \
    --latent-cluster-method umap \
    --model-selection final
done