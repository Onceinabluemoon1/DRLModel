import json
import os
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import absl.app
import absl.flags
import gym
import jaxrl_m.envs  # noqa: F401 — registers Gym env IDs via import side effects
import numpy as np
import torch
from jaxrl_m.envs.sort_easy import preference_training_adapter_from_gym_env
from tqdm import tqdm

import pref_learn.utils.plot_utils as putils
from pref_learn.configs.data_configs import get_train_vae_dataset
from pref_learn.models.mlp import (
    CategoricalModel,
    MeanVarianceModel,
    MLPClassifier,
    MLPModel,
)
from pref_learn.models.qlae import QLAE, build_qlae_optimizer
from pref_learn.models.tcvae import TCVAEModel
from pref_learn.models.utils import Annealer, EarlyStopper, get_datasets
from pref_learn.models.vae import VAEClassifier, VAEModel
from pref_learn.utils.disentangle_sweep import (
    enrich_variant_sweep_metadata,
    sweep_subruns_dir,
)
from pref_learn.utils.utils import (
    WandBLogger,
    define_flags_with_default,
    get_user_flags,
    prefix_metrics,
    set_random_seed,
)

FLAGS_DEF = define_flags_with_default(
    env="maze2d-target-v0",  # can change
    comment="",
    data_seed=42,
    batch_size=256,
    set_size=-1,
    early_stop=False,
    min_delta=3e-4,
    patience=10,
    lr=1e-3,
    model_type="MLP",  # can change
    # MLP
    hidden_dim=256,
    # Categorical
    num_atoms=10,
    r_min=0,
    r_max=1,
    entropy_coeff=0.1,
    # Mean Var
    variance_penalty=0.0,
    # VAE
    latent_dim=32,
    kl_weight=1.0,
    tc_weight=1.0,
    mi_weight=1.0,
    dwkl_weight=1.0,
    loss_aggregation="sum",
    normalize_kl=True,
    learned_prior=False,
    flow_prior=False,
    use_annealing=False,
    annealer_baseline=0.0,
    annealer_type="cosine",
    annealer_cycles=4,
    # QLAE (discrete latents)
    num_quantized_values=10,
    weight_decay=0.0,
    # Training
    n_epochs=500,
    eval_freq=50,
    save_freq=50,
    device="cuda",
    # Dataset
    pref_dataset="",
    dataset_path="",
    infomec_dataset_path="",
    logging=WandBLogger.get_default_config(),
    seed=42,
    # plotting
    debug_plots=False,
    plot_observations=False,
    reward_scaling=1.0,
    # biased
    biased_mode="grid",
    # If empty, `main` uses datetime.now(); set via env for sweeps, e.g. RUN_TIMESTAMP=... sbatch ...
    run_timestamp="",
    # e.g. latentsweep: shared sweep dir / W&B group for beta×latent_dim grids (see submit_train_vae_grid.bash)
    sweep_folder_tag="",
    # Decoder (VAE / TCVAE / QLAE); see make_reward_decoder in nets.py
    encoder_use_labels=True,  # pair encoder: concat y into pair_mlp input (q(z|s1,s2,y))
    decoder_type="mlp",  # mlp | linear | odinlinear | slotlinear | ellenlinear | offsetlinear
    decoder_gate_temperature=1.0,  # linear / slotlinear softmax: lower => sharper routing
    decoder_slot_routing="softmax",  # slotlinear only: softmax | sum
    decoder_orth_weight=0.0,  # odinlinear only: ODIN row orth on z_head.weight
)


def log_metrics(metrics, epoch, logger):
    for key, val in metrics.items():
        if isinstance(val, list):
            metrics[key] = np.mean(val)
    logger.log(metrics, step=epoch)


def resolve_dataset_flags(FLAGS) -> None:
    if FLAGS.pref_dataset.strip():
        cfg = get_train_vae_dataset(pref_dataset=FLAGS.pref_dataset.strip())
        FLAGS.env = cfg["env"]
        FLAGS.dataset_path = cfg["pref_dataset_path"]
        FLAGS.infomec_dataset_path = cfg["pref_infomec_dataset_path"]
        return

    assert FLAGS.dataset_path.strip(), "Provide --pref_dataset or --dataset_path."
    assert FLAGS.env.strip(), "Provide --pref_dataset or --env."


def effective_set_size(*, flags_set_size: int, dataset_path: str) -> int:
    if flags_set_size > 0:
        return int(flags_set_size)
    with open(dataset_path, "rb") as f:
        return int(pickle.load(f)["observations"].shape[1])


def _sweep_folder_display_tag(tag: str) -> str:
    tag = tag.strip()
    if tag == "latentgrid":
        return "lgrid"
    return tag


def _flag_bool(value: bool) -> str:
    return "T" if value else "F"


def _run_dir_timestamp(ts: str) -> str:
    ts = ts.strip()
    if ts.startswith("arrayjob_"):
        return ""
    return ts


def main(_):
    FLAGS = absl.flags.FLAGS
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if FLAGS.run_timestamp.strip():
        timestamp = FLAGS.run_timestamp.strip()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolve_dataset_flags(FLAGS)
    assert os.path.exists(FLAGS.dataset_path), f"Dataset not found: {FLAGS.dataset_path}"
    if FLAGS.infomec_dataset_path.strip():
        assert os.path.exists(FLAGS.infomec_dataset_path), (
            f"InfoMEC dataset not found: {FLAGS.infomec_dataset_path}"
        )
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    assert FLAGS.comment, "You must leave your comment for logging experiment."
    run_label = FLAGS.pref_dataset.strip() or FLAGS.env
    run_label = f"{run_label}_A={effective_set_size(flags_set_size=FLAGS.set_size, dataset_path=FLAGS.dataset_path)}"
    nz_tag = f"nz={FLAGS.latent_dim}"
    learned_prior_tag = f"learnpz={_flag_bool(FLAGS.learned_prior)}"
    run_tags = [nz_tag]
    if FLAGS.model_type != "QLAE":
        run_tags.append(learned_prior_tag)
    if "VAE" in FLAGS.model_type:
        run_tags.append(f"anneal={_flag_bool(FLAGS.use_annealing)}")
    if FLAGS.model_type == "QLAE":
        run_tags.append(f"V={FLAGS.num_quantized_values}")
        run_tags.append(f"wd={FLAGS.weight_decay}")
    dec_tag = f"dec={FLAGS.decoder_type}"
    run_tags.append(dec_tag)
    if str(FLAGS.decoder_type).lower() == "slotlinear":
        run_tags.append(f"slotr={FLAGS.decoder_slot_routing}")
    if str(FLAGS.decoder_type).lower() == "odinlinear":
        run_tags.append(f"orthw={FLAGS.decoder_orth_weight}")
    run_tags.append(f"encY={_flag_bool(FLAGS.encoder_use_labels)}")
    run_tag_str = "_".join(run_tags)
    sweep_folder_tag_raw = FLAGS.sweep_folder_tag.strip()
    sweep_folder_tag = (
        _sweep_folder_display_tag(tag=sweep_folder_tag_raw)
        if sweep_folder_tag_raw
        else ""
    )
    if sweep_folder_tag_raw:
        sweep_scope_tags = [sweep_folder_tag]
        if FLAGS.model_type != "QLAE":
            sweep_scope_tags.append(learned_prior_tag)
        if "VAE" in FLAGS.model_type:
            sweep_scope_tags.append(f"anneal={_flag_bool(FLAGS.use_annealing)}")
        if FLAGS.model_type == "QLAE":
            sweep_scope_tags.append(f"nv={FLAGS.num_quantized_values}")
        sweep_scope_tags.append(dec_tag)
        if str(FLAGS.decoder_type).lower() == "slotlinear":
            sweep_scope_tags.append(f"slotr={FLAGS.decoder_slot_routing}")
        if str(FLAGS.decoder_type).lower() == "odinlinear":
            sweep_scope_tags.append(f"orthw={FLAGS.decoder_orth_weight}")
        sweep_scope_tags.append(f"encY={_flag_bool(FLAGS.encoder_use_labels)}")
        sweep_scope_tag = "_".join(sweep_scope_tags)
    else:
        sweep_scope_tag = f"nz={FLAGS.latent_dim}"
    FLAGS.logging.experiment_id = (
        f"{run_label}_{FLAGS.model_type}_{run_tag_str}_{FLAGS.comment}"
    )
    if FLAGS.run_timestamp.strip():
        # Array sweeps: one W&B group per submit-time RUN_TIMESTAMP.
        if sweep_folder_tag:
            FLAGS.logging.group = (
                f"{run_label}_{FLAGS.model_type}_{sweep_scope_tag}_{timestamp}"
            )
        else:
            FLAGS.logging.group = (
                f"{run_label}_{FLAGS.model_type}_{run_tag_str}_{timestamp}"
            )
    else:
        FLAGS.logging.group = FLAGS.logging.experiment_id
    anneal_tag = ""
    if "VAE" in FLAGS.model_type:
        anneal_tag = f"_anneal={_flag_bool(FLAGS.use_annealing)}"
    ts_for_dir = _run_dir_timestamp(ts=timestamp)
    if sweep_folder_tag:
        run_subdir = f"{ts_for_dir}_{sweep_scope_tag}" if ts_for_dir else sweep_scope_tag
    else:
        run_subdir = (
            f"{ts_for_dir}_nz={FLAGS.latent_dim}_learnpz={_flag_bool(FLAGS.learned_prior)}_"
            f"agg={FLAGS.loss_aggregation}{anneal_tag}"
        )
    sweep_root = (
        Path(FLAGS.logging.output_dir) / run_label / str(FLAGS.model_type) / run_subdir
    )
    if FLAGS.run_timestamp.strip():
        save_dir = sweep_subruns_dir(sweep_root=sweep_root) / FLAGS.comment
    else:
        save_dir = sweep_root / FLAGS.comment
    FLAGS.logging.output_dir = str(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Per-epoch checkpoints (save_freq); best_model.pt stays in save_dir
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    gym_env = gym.make(FLAGS.env)
    gym_env.seed(FLAGS.seed)
    gym_env.action_space.seed(FLAGS.seed)
    gym_env.observation_space.seed(FLAGS.seed)
    reward_env = preference_training_adapter_from_gym_env(gym_env=gym_env)
    set_random_seed(FLAGS.seed)
    if hasattr(gym_env, "reward_observation_space"):
        obs_dim = gym_env.reward_observation_space.shape[0]
    else:
        obs_dim = gym_env.observation_space.shape[0]
    if "maze" in FLAGS.env:
        gym_env.set_biased_mode(FLAGS.biased_mode)
    action_dim = gym_env.action_space.shape[0]

    (
        train_loader,
        test_loader,
        train_dataset,
        eval_dataset,
        set_size,
        segment_len,
        obs_dim,
    ) = get_datasets(
        FLAGS.dataset_path,
        obs_dim,
        action_dim,
        FLAGS.batch_size,
        FLAGS.set_size,
    )

    if FLAGS.model_type == "MLP":
        reward_model = MLPModel(obs_dim, FLAGS.hidden_dim)
    elif FLAGS.model_type == "Categorical":
        reward_model = CategoricalModel(
            input_dim=obs_dim,
            hidden_dim=FLAGS.hidden_dim,
            n_atoms=FLAGS.num_atoms,
            r_min=FLAGS.r_min,
            r_max=FLAGS.r_max,
            entropy_coeff=FLAGS.entropy_coeff,
        )
    elif FLAGS.model_type == "MeanVar":
        reward_model = MeanVarianceModel(
            input_dim=obs_dim,
            hidden_dim=FLAGS.hidden_dim,
            variance_penalty=FLAGS.variance_penalty,
        )
    elif FLAGS.model_type == "TCVAE":
        annealer = None
        if FLAGS.use_annealing:
            annealer = Annealer(
                total_steps=FLAGS.n_epochs // FLAGS.annealer_cycles,
                shape=FLAGS.annealer_type,
                baseline=FLAGS.annealer_baseline,
                cyclical=FLAGS.annealer_cycles > 1,
            )
        reward_model = TCVAEModel(
            encoder_input=set_size * (2 * obs_dim * segment_len + 1),
            decoder_input=obs_dim + FLAGS.latent_dim,
            latent_dim=FLAGS.latent_dim,
            hidden_dim=FLAGS.hidden_dim,
            annotation_size=set_size,
            segment_len=segment_len,
            obs_dim=obs_dim,
            dataset_size=len(train_dataset),
            tc_weight=FLAGS.tc_weight,
            mi_weight=FLAGS.mi_weight,
            dwkl_weight=FLAGS.dwkl_weight,
            learned_prior=FLAGS.learned_prior,
            annealer=annealer,
            reward_scaling=FLAGS.reward_scaling,
            loss_aggregation=FLAGS.loss_aggregation,
            normalize_kl=FLAGS.normalize_kl,
            decoder_type=FLAGS.decoder_type,
            decoder_gate_temperature=FLAGS.decoder_gate_temperature,
            decoder_slot_routing=FLAGS.decoder_slot_routing,
            encoder_use_labels=FLAGS.encoder_use_labels,
            decoder_orth_weight=FLAGS.decoder_orth_weight,
        )
    elif FLAGS.model_type == "VAE" or FLAGS.model_type == "VAEClassifier":
        annealer = None
        if FLAGS.use_annealing:
            annealer = Annealer(
                total_steps=FLAGS.n_epochs // FLAGS.annealer_cycles,
                shape=FLAGS.annealer_type,
                baseline=FLAGS.annealer_baseline,
                cyclical=FLAGS.annealer_cycles > 1,
            )
        if FLAGS.model_type == "VAEClassifier":
            reward_model = VAEClassifier
            decoder_input = 2 * obs_dim + FLAGS.latent_dim
        else:
            reward_model = VAEModel
            decoder_input = obs_dim + FLAGS.latent_dim
        reward_model = reward_model(
            encoder_input=set_size * (2 * obs_dim * segment_len + 1),
            decoder_input=decoder_input,
            latent_dim=FLAGS.latent_dim,
            hidden_dim=FLAGS.hidden_dim,
            annotation_size=set_size,
            segment_len=segment_len,
            obs_dim=obs_dim,
            kl_weight=FLAGS.kl_weight,
            learned_prior=FLAGS.learned_prior,
            flow_prior=FLAGS.flow_prior,
            annealer=annealer,
            reward_scaling=FLAGS.reward_scaling,
            loss_aggregation=FLAGS.loss_aggregation,
            normalize_kl=FLAGS.normalize_kl,
            decoder_type=FLAGS.decoder_type,
            decoder_gate_temperature=FLAGS.decoder_gate_temperature,
            decoder_slot_routing=FLAGS.decoder_slot_routing,
            encoder_use_labels=FLAGS.encoder_use_labels,
            decoder_orth_weight=FLAGS.decoder_orth_weight,
        )
    elif FLAGS.model_type == "MLPClassifier":
        reward_model = MLPClassifier(obs_dim, FLAGS.hidden_dim)
    elif FLAGS.model_type == "QLAE":
        reward_model = QLAE(
            encoder_input=set_size * (2 * obs_dim * segment_len + 1),
            decoder_input=obs_dim + FLAGS.latent_dim,
            latent_dim=FLAGS.latent_dim,
            hidden_dim=FLAGS.hidden_dim,
            annotation_size=set_size,
            segment_len=segment_len,
            obs_dim=obs_dim,
            num_quantized_values=FLAGS.num_quantized_values,
            reward_scaling=FLAGS.reward_scaling,
            loss_aggregation=FLAGS.loss_aggregation,
            decoder_type=FLAGS.decoder_type,
            decoder_gate_temperature=FLAGS.decoder_gate_temperature,
            decoder_slot_routing=FLAGS.decoder_slot_routing,
            encoder_use_labels=FLAGS.encoder_use_labels,
            decoder_orth_weight=FLAGS.decoder_orth_weight,
        )
    else:
        raise NotImplementedError

    device = FLAGS.device
    reward_model = reward_model.to(device)
    variant["resolved_set_size"] = set_size
    variant["discrete_latents"] = bool(reward_model.discrete_latents)
    enrich_variant_sweep_metadata(variant=variant)
    with open(save_dir / "variant.json", "w", encoding="utf-8") as vf:
        json.dump(
            {"resolved_run_timestamp": timestamp, "variant": variant},
            vf,
            indent=2,
            sort_keys=True,
            default=str,
        )
    wb_logger = WandBLogger(FLAGS.logging, variant=variant)
    if FLAGS.model_type == "QLAE":
        optimizer = build_qlae_optimizer(
            model=reward_model,
            lr=FLAGS.lr,
            weight_decay=FLAGS.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(reward_model.parameters(), lr=FLAGS.lr)
    early_stop = EarlyStopper(FLAGS.patience, FLAGS.min_delta)
    best_criteria = None
    for epoch in tqdm(range(FLAGS.n_epochs), desc="epochs"):
        metrics = defaultdict(list)
        metrics["epoch"] = epoch

        for batch in train_loader:
            optimizer.zero_grad()
            observations = batch["observations"].to(device).float()
            observations_2 = batch["observations_2"].to(device).float()
            labels = batch["labels"].to(device).float()
            loss, batch_metrics = reward_model(observations, observations_2, labels)
            loss.backward()
            optimizer.step()

            for key, val in prefix_metrics(batch_metrics, "train").items():
                metrics[key].append(val)

        should_eval = (epoch % FLAGS.eval_freq == 0) or (epoch == FLAGS.n_epochs - 1)
        if should_eval:
            for batch in test_loader:
                with torch.no_grad():
                    observations = batch["observations"].to(device).float()
                    observations_2 = batch["observations_2"].to(device).float()
                    labels = batch["labels"].to(device).float()
                    loss, batch_metrics = reward_model(
                        observations, observations_2, labels
                    )

                    for key, val in prefix_metrics(batch_metrics, "eval").items():
                        metrics[key].append(val)

            if FLAGS.debug_plots and "maze2d" in FLAGS.env:
                if FLAGS.model_type == "MLP":
                    fig_dict = putils.plot_mlp(reward_env, reward_model)
                elif FLAGS.model_type == "Categorical" or FLAGS.model_type == "MeanVar":
                    fig_dict = putils.plot_mlp_samples(reward_env, reward_model)
                elif FLAGS.model_type == "MLPClassifier":
                    fig_dict = putils.plot_classifier(
                        reward_env, reward_model, eval_dataset
                    )
                else:
                    fig_dict = putils.plot_vae(
                        reward_env,
                        reward_model,
                        eval_dataset,
                        classifier="Classifier" in FLAGS.model_type,
                    )
                metrics.update(prefix_metrics(fig_dict, "debug_plots"))
            elif "VAE" in FLAGS.model_type or FLAGS.model_type == "QLAE":
                putils.update_posterior(reward_env, reward_model, eval_dataset)

            criteria = np.mean(metrics["eval/total_loss"])

            if best_criteria is None:
                best_criteria = criteria
                torch.save(reward_model, save_dir / "best_model.pt")

            if criteria < best_criteria:
                torch.save(reward_model, save_dir / "best_model.pt")
                best_criteria = criteria

            if FLAGS.early_stop and early_stop.early_stop(criteria):
                log_metrics(metrics, epoch, wb_logger)
                torch.save(reward_model, save_dir / "checkpoints" / f"model_{epoch}.pt")
                break

        should_save = (epoch % FLAGS.save_freq == 0) or (epoch == FLAGS.n_epochs - 1)
        if should_save:
            torch.save(reward_model, save_dir / "checkpoints" / f"model_{epoch}.pt")

        if FLAGS.use_annealing and reward_model.annealer is not None:
            reward_model.annealer.step()

        log_metrics(metrics, epoch, wb_logger)


if __name__ == "__main__":
    absl.app.run(main)
