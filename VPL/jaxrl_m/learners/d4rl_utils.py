import d4rl
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxrl_m.dataset import Dataset
from tqdm import tqdm


def plot_observation_rewards(obs, r):
    fig, ax = plt.subplots()
    sc = ax.scatter(obs[:, 0], obs[:, 1], c=r)
    plt.colorbar(sc, ax=ax)
    plt.close(fig)
    return fig


def make_env(env_name: str):
    env = gym.make(env_name)
    return env


def expand_dataset(dataset, ratio=1):
    ratio = int(ratio)
    for k, v in dataset.items():
        dataset[k] = np.concatenate([dataset[k]] * ratio)
    return dataset


def full_horizon_qlearning_dataset(env, dataset=None, **kwargs):
    """Q-learning layout keeping every env step, including each episode terminal."""
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    obs = dataset["observations"].astype(np.float32)
    actions = dataset["actions"].astype(np.float32)
    rewards = dataset["rewards"].astype(np.float32)
    terminals = dataset["terminals"].astype(bool)

    next_obs = np.empty_like(obs)
    next_obs[:-1] = obs[1:]
    next_obs[-1] = obs[-1]
    next_obs[terminals] = obs[terminals]

    terminals_f = terminals.astype(np.float32)
    return {
        "observations": obs,
        "actions": actions,
        "next_observations": next_obs,
        "rewards": rewards,
        "terminals": terminals_f,
        "traj_done": terminals_f.copy(),
    }


def get_dataset(
    env: gym.Env,
    clip_to_eps: bool = True,
    eps: float = 1e-5,
    append_goal: bool = False,
    use_reward_model: bool = False,
    model_type: str = "",
    reward_model: torch.nn.Module = None,
    fix_mode: int = -1,
    terminate_on_end: bool = False,
    comp_size: int = 1000,
    vae_norm: str = "fixed",
    vae_sampling: bool = False,
    sample_freq: int = 10,
    next_step: bool = False,
    exp_scaling_temp: float = 1.0,
):
    if terminate_on_end:
        dataset = full_horizon_qlearning_dataset(env)
    else:
        dataset = d4rl.qlearning_dataset(env, terminate_on_end=False)
    # if comp_size < 10:
    dataset = expand_dataset(dataset)
    # comp_size = 1000

    dones_float = dataset["terminals"].astype(np.float32)
    dataset = {
        "observations": dataset["observations"],
        "actions": dataset["actions"],
        "rewards": dataset["rewards"],
        "masks": 1.0 - dataset["terminals"],
        "dones_float": dones_float,
        "next_observations": dataset["next_observations"],
        "traj_done": dataset["traj_done"],
    }
    comparison_obs = None
    if use_reward_model:
        assert reward_model is not None
        dataset, comparison_obs = relabel_rewards_with_model(
            env,
            dataset,
            model_type,
            reward_model,
            append_goal,
            fix_mode,
            comp_size,
            vae_norm,
            vae_sampling,
            sample_freq,
            next_step,
            exp_scaling_temp,
        )
    elif hasattr(env, "relabel_offline_reward") and env.relabel_offline_reward:
        dataset = relabel_rewards_with_env(env, dataset, append_goal)

    dataset = {k: v.astype(np.float32) for k, v in dataset.items()}
    print(f"Dataset ({env.spec.id}) obs shape: {dataset['observations'].shape}")
    return Dataset(dataset), comparison_obs


def new_get_trj_idx(dataset):
    N = dataset["rewards"].shape[0]
    if "traj_done" in dataset and len(dataset["traj_done"]) == N:
        dones_float = dataset["traj_done"]
    elif "dones_float" in dataset:
        dones_float = dataset["dones_float"]
    else:
        dones_float = dataset["dones_float"]
    # If the dones are dropped just split uniformly to relabel the trajectories
    if len(np.argwhere(dones_float)) < 2:
        print("Dones are dropped, splitting uniformly")
        idx = np.arange(0, len(dones_float), len(dones_float) // 50000)
        dones_float[idx] = 1.0
        dones_float[-1] = 1.0
    else:
        print("Dones are not dropped")

    N = dataset["rewards"].shape[0]
    episode_step = 0
    start_idx, data_idx = 0, 0
    trj_idx_list = []
    for i in tqdm(range(N - 1), desc="Getting trj idx"):
        done_bool = dones_float[i]
        if done_bool:
            episode_step = 0
            trj_idx_list.append([start_idx, data_idx])
            start_idx = data_idx + 1

        episode_step += 1
        data_idx += 1

    trj_idx_list.append([start_idx, data_idx])
    return trj_idx_list


def sample_comparison_states(observations, size, obs_size):
    indices = np.random.choice(observations.shape[0], size, replace=False)
    return observations[indices, :obs_size]


def relabel_rewards_with_model(
    env,
    dataset,
    model_type,
    reward_model,
    append_goal,
    fix_mode,
    comp_size=1000,
    vae_norm="fixed",
    vae_sampling=False,
    sample_freq=10,
    next_step=False,
    exp_scaling_temp=1.0,
):
    observation_dim = env.observation_space.shape[-1]
    if hasattr(env, "reward_observation_space"):
        observation_dim = env.reward_observation_space.shape[-1]

    obs_list = []
    next_obs_list = []
    new_rewards = np.zeros_like(dataset["rewards"])
    new_next_rewards = np.zeros_like(dataset["rewards"])
    mode_mask = np.zeros_like(dataset["rewards"])
    traj_idx = new_get_trj_idx(dataset)

    sampled_z = None
    sample_every = 10
    comparison_obs = None
    if "Classifier" in model_type or vae_norm in ("learned_norm", "prob"):
        comparison_obs = sample_comparison_states(
            observations=dataset["observations"],
            size=comp_size,
            obs_size=observation_dim,
        )
        comparison_obs = (
            torch.from_numpy(comparison_obs)
            .float()
            .to(next(reward_model.parameters()).device)
        )
    for traj_id, (start, end) in enumerate(
        tqdm(
            traj_idx,
            desc=f"Relabel {env.spec.id} {model_type}, ag: {append_goal}, fm: {fix_mode}",
        )
    ):
        obs = dataset["observations"][start : end + 1]
        next_obs = dataset["next_observations"][start : end + 1]

        input_obs = (
            torch.from_numpy(obs[:, :observation_dim])
            .float()
            .to(next(reward_model.parameters()).device)
        )

        input_next_obs = (
            torch.from_numpy(next_obs[:, :observation_dim])
            .float()
            .to(next(reward_model.parameters()).device)
        )
        idx = fix_mode
        if model_type == "MLP":
            with torch.no_grad():
                rewards = reward_model.get_reward(input_obs)
                next_rewards = reward_model.get_reward(input_next_obs)
        elif model_type == "MLPClassifier":
            with torch.no_grad():
                rewards = reward_model.get_reward(input_obs, comparison_obs)
                next_rewards = reward_model.get_reward(input_next_obs, comparison_obs)
        elif model_type == "Categorical" or model_type == "MeanVar":
            with torch.no_grad():
                rewards = reward_model.sample_reward(input_obs)
                next_rewards = reward_model.sample_reward(input_next_obs)
        elif model_type == "VAE" or model_type == "TCVAE":
            with torch.no_grad():
                if vae_sampling:
                    if sampled_z is None or traj_id % sample_every == 0:
                        sampled_z = sample_latent(reward_model, env)
                        if vae_norm == "learned_norm":
                            norm_z = get_norm(
                                traj_id, sampled_z, env, reward_model, comparison_obs
                            )
                    z = torch.tensor(sampled_z)
                else:
                    if fix_mode < 0:
                        idx = env.sample_mode()
                    else:
                        idx = fix_mode
                    z = torch.tensor(reward_model.biased_latents[idx])
                batch_z = (
                    z.repeat(obs.shape[0], 1)
                    .float()
                    .to(next(reward_model.parameters()).device)
                )
                if vae_norm == "prob":
                    assert comparison_obs is not None
                    rewards = reward_model.get_prob_reward(
                        s_a=input_obs,
                        z=batch_z,
                        comparison_obs=comparison_obs,
                    )
                    next_rewards = reward_model.get_prob_reward(
                        s_a=input_next_obs,
                        z=batch_z,
                        comparison_obs=comparison_obs,
                    )
                else:
                    rewards = reward_model.get_reward(
                        torch.cat([input_obs, batch_z], dim=-1)
                    )
                    next_rewards = reward_model.get_reward(
                        torch.cat([input_next_obs, batch_z], dim=-1)
                    )

                    if vae_norm == "learned_norm":
                        rewards = torch.exp((rewards - norm_z) * 1e-3)

                if append_goal:
                    if vae_sampling:
                        raise NotImplementedError
                    # Changing the latent
                    goal = np.array(env.goals[idx, :2])
                    batch_z = torch.ones_like(batch_z[:, :2]).cpu().numpy() * goal
                    # Appending the task vector to the observation
                    obs_list.append(np.concatenate([obs, batch_z], axis=-1))
                    next_obs_list.append(np.concatenate([next_obs, batch_z], axis=-1))
                else:
                    # Appending the task vector to the observation
                    obs_list.append(np.concatenate([obs, batch_z.cpu().numpy()], axis=-1))
                    next_obs_list.append(
                        np.concatenate([next_obs, batch_z.cpu().numpy()], axis=-1)
                    )
        new_rewards[start : end + 1] = rewards.squeeze().cpu().numpy()
        new_next_rewards[start : end + 1] = next_rewards.squeeze().cpu().numpy()
        mode_mask[start : end + 1] = idx

    if len(obs_list) > 0:
        dataset["observations"] = np.concatenate(obs_list, axis=0)
        dataset["next_observations"] = np.concatenate(next_obs_list, axis=0)

    if model_type == "MLP" or model_type == "Categorical" or model_type == "MeanVar":
        print("Baseline models")
        new_rewards = (new_rewards - new_rewards.min()) / (
            new_rewards.max() - new_rewards.min()
        )
        new_rewards = np.exp(new_rewards / 0.1)
        new_rewards = new_rewards * 1e-2
        dataset["rewards"] = new_rewards
        dataset["next_rewards"] = new_next_rewards
    elif (model_type == "VAE" or model_type == "TCVAE") and vae_norm == "fixed":
        print("VAE with fixed norm")
        for mode in range(env.get_num_modes()):
            id_r = np.argwhere(mode_mask == mode)
            new_rewards[id_r] = (new_rewards[id_r] - new_rewards[id_r].min()) / (
                new_rewards[id_r].max() - new_rewards[id_r].min()
            )
        new_rewards = new_rewards * 1e-2
        dataset["rewards"] = new_rewards
        dataset["next_rewards"] = new_next_rewards
    elif (model_type == "VAE" or model_type == "TCVAE") and vae_norm == "prob":
        print("VAE with prob (BTL expected preference) scaling")
        dataset["rewards"] = new_rewards.squeeze()
        dataset["next_rewards"] = new_next_rewards.squeeze()
    elif vae_norm == "max":
        dataset["rewards"] = new_rewards / np.abs(new_rewards).max()
        dataset["next_rewards"] = new_next_rewards / np.abs(new_next_rewards).max()
    else:
        print("VAE with learned norm or no norm")
        dataset["rewards"] = new_rewards
        dataset["next_rewards"] = new_next_rewards

    if next_step:
        dataset["rewards"] = dataset["next_rewards"]

    return dataset, comparison_obs


def relabel_rewards_with_env(env, dataset, append_goal):
    new_rewards = []
    new_next_rewards = []
    obs_list = []
    next_obs_list = []
    traj_idx = new_get_trj_idx(dataset)
    modes = []
    for start, end in tqdm(
        traj_idx,
        desc=f"Relabelling reward with {env.spec.id}, append_goal: {append_goal}",
    ):
        obs = dataset["observations"][start : end + 1]
        next_obs = dataset["next_observations"][start : end + 1]
        mode = env.sample_mode()
        new_rewards.append(env.compute_reward(obs[None], mode)[0])
        new_next_rewards.append(env.compute_reward(next_obs[None], mode)[0])
        modes.append(mode)
        if append_goal:
            goal = np.array(env.goals[mode, :2])
            obs_list.append(
                np.concatenate([obs, np.ones_like(obs[:, :2]) * goal], axis=-1)
            )
            next_obs_list.append(
                np.concatenate([next_obs, np.ones_like(next_obs[:, :2]) * goal], axis=-1)
            )
    modes_arr = np.asarray(modes, dtype=np.int64)
    uniq_modes, mode_counts = np.unique(modes_arr, return_counts=True)
    n_tot = int(modes_arr.size)
    print("Modes during relabel_rewards_with_env:")
    for m, c in zip(uniq_modes, mode_counts):
        pct = 100.0 * float(c) / float(n_tot)
        print(f"  traj mode {int(m):>4}: count {int(c):>8} ({pct:6.2f}%)")
    print(f"  total:       {n_tot:>8}")
    print(f"  mean mode:   {float(np.mean(modes_arr)):8.4f}")

    new_rewards = np.concatenate(new_rewards)
    new_next_rewards = np.concatenate(new_next_rewards)
    dataset["rewards"] = new_rewards
    dataset["next_rewards"] = new_next_rewards
    if append_goal:
        dataset["observations"] = np.concatenate(obs_list, axis=0)
        dataset["next_observations"] = np.concatenate(next_obs_list, axis=0)
    return dataset


def sample_latent(reward_model, env):
    z_orig = reward_model.sample_prior(1)
    return z_orig


def get_norm(i, z, env, reward_model, obs):
    # obs = torch.from_numpy(obs).float().to(next(reward_model.parameters()).device)
    z = z.repeat(obs.shape[0], 1).float().to(next(reward_model.parameters()).device)
    r = reward_model.decode(obs, z).view(-1, 1)  # .reshape((NX, NY))
    N = r.shape[0]
    norm_z = (torch.logsumexp(r, dim=0) - np.log(N)).item()
    return norm_z
