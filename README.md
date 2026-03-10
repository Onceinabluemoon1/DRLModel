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

