# Disentangled Reward Learning


## Dependencies

install with conda environment yaml

```bash
conda env create -f environment.yml
```

activate conda
```bash
conda activate dlr
```

## Synthetic data creation

For instructions on generating synthetic data, see:
- [synthetic_data/README.md](synthetic_data/README.md)
# Disentangled Reward Learning

## Dependencies

install with conda environment yaml

```bash
conda env create -f environment.yml
```

activate conda

```bash
conda activate drl
```

## Synthetic data creation

For instructions on generating synthetic data, see:

* [synthetic_data/README.md](synthetic_data/README.md)

## Hopper / UniRLHF trajectory pipeline

This repo includes a Hopper pairwise preference pipeline built from:

* D4RL Hopper trajectories
* UniRLHF human pairwise preference labels

The overall flow is:

1. install the repo environment
2. install D4RL from source
3. download / access the Hopper D4RL dataset
4. place the UniRLHF Hopper human-label `.pkl` files under `uni_rlhf/data/`
5. build the cleaned pairwise trajectory dataset
6. train the BT baseline
7. train the Hopper VAE

## D4RL installation

D4RL should be installed from source.

```bash
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip install -e .
cd ..
```

D4RL datasets are downloaded automatically when `env.get_dataset()` is called. By default they are stored under:

```text
~/.d4rl/datasets
```

For MuJoCo locomotion environments such as Hopper, D4RL also requires MuJoCo / `mujoco_py` support. If Hopper environments fail to import, make sure the MuJoCo-side dependencies are installed correctly for your system.

## Download / access the Hopper trajectories

After D4RL is installed, run:

```bash
python - <<'PY'
import gym
import d4rl

env = gym.make("hopper-medium-v2")
dataset = env.get_dataset()

print(dataset.keys())
print(dataset["observations"].shape)
print(dataset["actions"].shape)
PY
```

This should download and cache the Hopper dataset and make the trajectories available through D4RL.

For the Hopper pairwise pipeline in this repo, the expected D4RL trajectory file is:

```text
~/.d4rl/datasets/hopper_medium-v2.hdf5
```

## Prepare the UniRLHF Hopper label files

Place the Hopper human-label files under:

```text
uni_rlhf/data/hopper-medium-v2_human_labels/
```

The directory should contain three files:

* `indices_1_*.pkl`
* `indices_2_*.pkl`
* `human_label_*.pkl`

These files define pairwise comparisons over fixed-length Hopper trajectory chunks:

* `indices_1[i]` = start index of trajectory segment 1
* `indices_2[i]` = start index of trajectory segment 2
* `human_label[i]` = human preference label for that pair

For the current Hopper setup, the compared chunks are fixed-length segments of length 200.

## Build the cleaned Hopper pairwise dataset

This script loads the D4RL Hopper trajectories and UniRLHF label files, aligns the comparison indices with the trajectory buffer, removes invalid or boundary-crossing comparisons, drops `-1` labels by default, and writes a cleaned `.npz` dataset for training.

Run:

```bash
python -m uni_rlhf.scripts.build_hopper_pair_dataset
```

You can also provide paths explicitly:

```bash
python -m uni_rlhf.scripts.build_hopper_pair_dataset \
  --hdf5-path ~/.d4rl/datasets/hopper_medium-v2.hdf5 \
  --label-dir uni_rlhf/data/hopper-medium-v2_human_labels \
  --output-path uni_rlhf/data/processed/hopper_medium_v2_pairwise_clean.npz
```

By default, the cleaned dataset is written to:

```text
uni_rlhf/data/processed/hopper_medium_v2_pairwise_clean.npz
```

The cleaned dataset contains:

* `x1_obs`
* `x1_act`
* `x2_obs`
* `x2_act`
* `y`
* `x1_return`
* `x2_return`
* `indices_1`
* `indices_2`

For the current Hopper setup:

* `T = 200`
* `obs_dim = 11`
* `act_dim = 3`

## Train the BT baseline

The BT baseline is the first sanity-check model. It uses a TCN over state sequences, a TCN over action sequences, and a Bradley-Terry preference objective.

Run:

```bash
python -m uni_rlhf.scripts.train_bt_baseline --normalize-obs --normalize-act
```

By default, outputs are written to:

```text
uni_rlhf/outputs/bt_baseline_hopper_medium_v2/
```

## Train the Hopper VAE

The Hopper VAE is the first full pairwise trajectory-reconstruction VAE for this setup.

Encoder:

* TCN over state trajectories
* TCN over action trajectories
* MLP over the preference label

Decoder reconstructs:

* first trajectory observations
* first trajectory actions
* second trajectory observations
* second trajectory actions
* preference label

Training objective:

* observation reconstruction loss
* action reconstruction loss
* preference reconstruction loss
* KL regularization

Run:

```bash
python -m uni_rlhf.scripts.train_hopper_vae --normalize-obs --normalize-act
```

You can also override key training settings, for example:

```bash
python -m uni_rlhf.scripts.train_hopper_vae \
  --normalize-obs \
  --normalize-act \
  --batch-size 8 \
  --hidden-dim 64 \
  --latent-dim 16
```

By default, outputs are written to:

```text
uni_rlhf/outputs/hopper_vae_medium_v2/
```

## Script summary

From zero to running on trajectories:

```bash
conda env create -f environment.yml
conda activate drl

git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip install -e .
cd ..

python - <<'PY'
import gym
import d4rl

env = gym.make("hopper-medium-v2")
dataset = env.get_dataset()
print(dataset["observations"].shape)
print(dataset["actions"].shape)
PY

python -m uni_rlhf.scripts.build_hopper_pair_dataset
python -m uni_rlhf.scripts.train_bt_baseline --normalize-obs --normalize-act
python -m uni_rlhf.scripts.train_hopper_vae --normalize-obs --normalize-act
```

## Notes

* The Hopper comparisons are already chunked into fixed-length segments, so no extra chunking is required for the current Hopper setup.
* The cleaned dataset builder drops `-1` labels by default, since those are treated as ambiguous / tie-like labels for the current pipeline.
* The builder also removes comparisons whose fixed-length windows cross episode boundaries.
* For the current Hopper setup, all model training is done on fixed-length 200-step segments.

## VPL

This is easier with conda.

## Conda packages

```bash
conda install -n vpl -c conda-forge numpy=1.22.4 -y
conda install -n vpl -c conda-forge scipy=1.11.4 -y
conda install -n vpl -c conda-forge scikit-learn=1.3.2 -y
conda install -n vpl -c conda-forge h5py=3.10.0 -y
conda install -n vpl -c conda-forge matplotlib=3.7.4 -y
conda install -n vpl -c conda-forge scikit-image=0.21.0 -y
conda install -n vpl -c conda-forge opencv=4.9 -y
conda install -n vpl -c conda-forge pybullet -y
conda install -n vpl -c conda-forge transforms3d -y
conda install -n vpl -c conda-forge meshcat -y
conda install -n vpl -c conda-forge pillow=10.0.1 -y
conda install -n vpl -c conda-forge pyyaml=6.0.1 -y
conda install -n vpl -c conda-forge tqdm=4.66.1 -y
```

### Pip packages

```bash
pip install jax==0.4.21
pip install jaxlib==0.4.21
pip install flax==0.7.2
pip install optax==0.1.7
pip install chex==0.1.7
pip install distrax==0.1.3
pip install ml-dtypes==0.3.1
pip install absl-py==2.0.0
pip install ml-collections==0.1.1
pip install cloudpickle==3.0.0
pip install gym==0.23.1
pip install timeout-decorator==0.5.0
pip install wandb==0.16.0
pip install protobuf==4.25.0
pip install imageio==2.32.0
pip install -e ./VPL/dependencies/d4rl --no-deps
```

## Required data
The Raven offline dataset should exist at:
VPL/jaxrl_m/envs/dataset/sort.hdf5

```bash
conda activate vpl
export PYTHONPATH=$PWD/VPL:$PWD/VPL/dependencies/ravens:$PYTHONPATH
export WANDB_MODE=offline

python -m pref_learn.create_dataset \
  --env=sort-easy-v0 \
  --data_dir=./VPL/pref_datasets \
  --num_query=5000 \
  --query_len=1 \
  --set_len=16

python VPL/pref_learn/train.py \
  --comment=sort_final_reward_results \
  --env=sort-easy-v0 \
  --dataset_path=./VPL/pref_datasets/sort-easy-v0/relabelled_queries_num5000_q1_s16 \
  --model_type=VAE \
  --logging.output_dir=./VPL/logs3 \
  --seed=42 \
  --learned_prior=True \
  --use_annealing=True \
  --n_epochs=500

python VPL/scripts/extract_raven_latents.py
```