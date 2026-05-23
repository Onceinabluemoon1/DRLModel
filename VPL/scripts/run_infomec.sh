#* run for a single training saveout (variant.json + best_model.pt); outputs go in the same dir
python scripts/plot_infomec.py \
  --input_dir "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_2goals_T=1/VAE/20260522_204730_ld=4_learned-prior=True/06_beta=3.0" \
  --no-latent-cluster

#*  run for a beta sweep (each value = training directory; per-run artifacts live there; summary under --outdir)
python scripts/plot_infomec_sweep.py \
  --input_dir "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/outputs/sort_4goals_T=10/VAE/20260518_154659_ld=4_learned-prior=False" \
  --latent-cluster-method tsne

