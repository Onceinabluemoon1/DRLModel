seed=-1
n_queries=20000
set_size=16
segment_len=1


# create pref dataset for 4 goals task
env="sort-easy-4goals-v0"
h5_fp="/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-4goals-v0/20260517_212722_ntrajs=5000/rollout_dataset.hdf5"
python -m pref_learn.create_dataset_infomec \
  --env="${env}" \
  --num_query="${n_queries}" \
  --segment_len="${segment_len}" \
  --set_size="${set_size}" \
  --offline_dataset_h5="${h5_fp}" \
  --store_source_mode=True \
  --data_seed="${seed}"

python -m pref_learn.create_dataset_infomec \
  --env="${env}" \
  --num_query="${n_queries}" \
  --segment_len="${segment_len}" \
  --set_size="${set_size}" \
  --offline_dataset_h5="${h5_fp}" \
  --store_source_mode=True \
  --data_seed="${seed}"


# # create pref dataset for 2 goals task
# env="sort-easy-2goals-v0"
# h5_fp="/scr/yutaizho/code/p-disentangle/Disentangled_Reward_Learning/data/ravens/sort-easy-2goals-v0/_vpl_official/rollout_dataset.hdf5"
# python -m pref_learn.create_dataset_infomec \
#   --env="${env}" \
#   --num_query="${n_queries}" \
#   --segment_len="${segment_len}" \
#   --set_size="${set_size}" \
#   --offline_dataset_h5="${h5_fp}" \
#   --store_source_mode=True \
#   --data_seed="${seed}"

