from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from uni_rlhf.src.datasets.hopper_pair_dataset import (
    HopperPairDataset,
    hopper_pair_collate_fn,
)
from uni_rlhf.src.models.hopper_vae import (
    HopperPairVAE,
    HopperVAEConfig,
    kl_divergence_gaussian,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    dataset_path: str
    output_dir: str
    seed: int = 42
    batch_size: int = 64
    num_epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    hidden_dim: int = 128
    num_tcn_blocks: int = 4
    kernel_size: int = 5
    dropout: float = 0.1
    latent_dim: int = 32
    label_embed_dim: int = 16
    label_hidden_dim: int = 64
    pref_hidden_dim: int = 128
    normalize_obs: bool = True
    normalize_act: bool = True
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # loss weights
    obs_loss_weight: float = 1.0
    act_loss_weight: float = 1.0
    pref_loss_weight: float = 1.0
    beta_max: float = 0.05
    beta_warmup_epochs: int = 10


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Hopper pairwise VAE.")
    parser.add_argument(
        "--dataset-path",
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-tcn-blocks", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--label-embed-dim", type=int, default=16)
    parser.add_argument("--label-hidden-dim", type=int, default=64)
    parser.add_argument("--pref-hidden-dim", type=int, default=128)
    parser.add_argument("--normalize-obs", action="store_true")
    parser.add_argument("--normalize-act", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--obs-loss-weight", type=float, default=1.0)
    parser.add_argument("--act-loss-weight", type=float, default=1.0)
    parser.add_argument("--pref-loss-weight", type=float, default=1.0)
    parser.add_argument("--beta-max", type=float, default=0.05)
    parser.add_argument("--beta-warmup-epochs", type=int, default=10)

    args = parser.parse_args()

    return TrainConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        hidden_dim=args.hidden_dim,
        num_tcn_blocks=args.num_tcn_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
        label_embed_dim=args.label_embed_dim,
        label_hidden_dim=args.label_hidden_dim,
        pref_hidden_dim=args.pref_hidden_dim,
        normalize_obs=args.normalize_obs,
        normalize_act=args.normalize_act,
        num_workers=args.num_workers,
        device=args.device,
        obs_loss_weight=args.obs_loss_weight,
        act_loss_weight=args.act_loss_weight,
        pref_loss_weight=args.pref_loss_weight,
        beta_max=args.beta_max,
        beta_warmup_epochs=args.beta_warmup_epochs,
    )


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_loaders(config: TrainConfig) -> Tuple[DataLoader, DataLoader, HopperPairDataset]:
    dataset = HopperPairDataset(
        npz_path=config.dataset_path,
        normalize_obs=config.normalize_obs,
        normalize_act=config.normalize_act,
    )

    n_total = len(dataset)
    n_val = max(1, int(config.val_fraction * n_total))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=hopper_pair_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=hopper_pair_collate_fn,
    )
    return train_loader, val_loader, dataset


def get_beta(epoch: int, beta_max: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return beta_max
    progress = min(epoch / warmup_epochs, 1.0)
    return beta_max * progress


def compute_metrics(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    obs_loss = (
        F.mse_loss(outputs["x1_obs_recon"], batch["x1_obs"]) +
        F.mse_loss(outputs["x2_obs_recon"], batch["x2_obs"])
    )

    act_loss = (
        F.mse_loss(outputs["x1_act_recon"], batch["x1_act"]) +
        F.mse_loss(outputs["x2_act_recon"], batch["x2_act"])
    )

    pref_loss = F.binary_cross_entropy_with_logits(
        outputs["pref_logit"],
        batch["y"].float(),
    )

    kl = kl_divergence_gaussian(outputs["mu"], outputs["logvar"])

    pref_acc = ((outputs["pref_logit"] >= 0).long() == batch["y"]).float().mean()

    return {
        "obs_loss": obs_loss,
        "act_loss": act_loss,
        "pref_loss": pref_loss,
        "kl": kl,
        "pref_acc": pref_acc,
    }


def run_epoch(
    model: HopperPairVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    beta = get_beta(epoch, config.beta_max, config.beta_warmup_epochs)

    total = {
        "loss": 0.0,
        "obs_loss": 0.0,
        "act_loss": 0.0,
        "pref_loss": 0.0,
        "kl": 0.0,
        "pref_acc": 0.0,
    }
    total_n = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(batch)
        metrics = compute_metrics(outputs, batch)

        loss = (
            config.obs_loss_weight * metrics["obs_loss"] +
            config.act_loss_weight * metrics["act_loss"] +
            config.pref_loss_weight * metrics["pref_loss"] +
            beta * metrics["kl"]
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bsz = batch["y"].shape[0]
        total["loss"] += float(loss.item()) * bsz
        total["obs_loss"] += float(metrics["obs_loss"].item()) * bsz
        total["act_loss"] += float(metrics["act_loss"].item()) * bsz
        total["pref_loss"] += float(metrics["pref_loss"].item()) * bsz
        total["kl"] += float(metrics["kl"].item()) * bsz
        total["pref_acc"] += float(metrics["pref_acc"].item()) * bsz
        total_n += bsz

    out = {k: v / max(total_n, 1) for k, v in total.items()}
    out["beta"] = beta
    return out


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, dataset = build_loaders(config)
    device = torch.device(config.device)

    model_config = HopperVAEConfig(
        obs_dim=dataset.obs_dim,
        act_dim=dataset.act_dim,
        seq_len=dataset.seg_len,
        hidden_dim=config.hidden_dim,
        num_tcn_blocks=config.num_tcn_blocks,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        label_embed_dim=config.label_embed_dim,
        label_hidden_dim=config.label_hidden_dim,
        latent_dim=config.latent_dim,
        num_label_classes=2,
        pref_hidden_dim=config.pref_hidden_dim,
    )
    model = HopperPairVAE(model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    config_dict = asdict(config)
    config_dict["dataset_stats"] = dataset.get_stats()
    config_dict["model_config"] = asdict(model_config)
    save_json(output_dir / "config.json", config_dict)

    print("=== DATASET STATS ===")
    print(json.dumps(dataset.get_stats(), indent=2))
    print("\n=== MODEL CONFIG ===")
    print(json.dumps(asdict(model_config), indent=2))
    print("\n=== TRAINING ===")

    best_val_score = math.inf
    history = []

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, epoch, config)
        val_metrics = run_epoch(model, val_loader, None, device, epoch, config)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_obs_loss": train_metrics["obs_loss"],
            "train_act_loss": train_metrics["act_loss"],
            "train_pref_loss": train_metrics["pref_loss"],
            "train_kl": train_metrics["kl"],
            "train_pref_acc": train_metrics["pref_acc"],
            "val_loss": val_metrics["loss"],
            "val_obs_loss": val_metrics["obs_loss"],
            "val_act_loss": val_metrics["act_loss"],
            "val_pref_loss": val_metrics["pref_loss"],
            "val_kl": val_metrics["kl"],
            "val_pref_acc": val_metrics["pref_acc"],
            "beta": train_metrics["beta"],
        }
        history.append(row)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.4f} "
            f"obs={row['train_obs_loss']:.4f} "
            f"act={row['train_act_loss']:.4f} "
            f"pref={row['train_pref_loss']:.4f} "
            f"kl={row['train_kl']:.4f} "
            f"acc={row['train_pref_acc']:.4f} | "
            f"val_loss={row['val_loss']:.4f} "
            f"obs={row['val_obs_loss']:.4f} "
            f"act={row['val_act_loss']:.4f} "
            f"pref={row['val_pref_loss']:.4f} "
            f"kl={row['val_kl']:.4f} "
            f"acc={row['val_pref_acc']:.4f} "
            f"beta={row['beta']:.4f}"
        )

        if row["val_loss"] < best_val_score:
            best_val_score = row["val_loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config_dict,
                    "epoch": epoch,
                    "val_loss": best_val_score,
                },
                output_dir / "best_model.pt",
            )

    save_json(output_dir / "history.json", {"history": history})
    print(f"\nBest val loss: {best_val_score:.4f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()