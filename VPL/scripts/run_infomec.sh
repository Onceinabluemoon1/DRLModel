#* run for a single training saveout (variant.json + best_model.pt); outputs go in the same dir
python scripts/plot_infomec.py \
  --run-dir "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL/logs3/sort-easy-v0/VAEClassifier/20260424_214120/0_beta=0.01_ld=8" \
  --split eval


#*  run for a beta sweep (each value = training directory; per-run artifacts live there; summary under --outdir)
python scripts/plot_infomec_sweep.py \
  --sweep-root "/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/VPL/logs3/sort-easy-v0/VAEClassifier/20260424_214120" \
  --split eval

