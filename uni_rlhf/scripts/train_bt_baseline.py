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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from uni_rlhf.src.datasets.hopper_pair_dataset import (
    HopperPairDataset,
    hopper_pair_collate_fn,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = x

        out = self.conv1(x)  # (B, C, T)
        out = out.transpose(1, 2)  # (B, T, C)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)  # (B, C, T)

        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)

        return residual + out


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        dilations = [2**i for i in range(num_blocks)]
        self.blocks = nn.ModuleList(
            [
                TCNBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.input_proj(x)  # (B, H, T)

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)  # (B, T, H)
        x = self.out_norm(x)
        x = x.mean(dim=1)  # mean pool over time -> (B, H)
        return x


class PairwiseBTModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        num_tcn_blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
        score_hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.state_encoder = TemporalEncoder(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_tcn_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.action_encoder = TemporalEncoder(
            input_dim=act_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_tcn_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        pair_dim = hidden_dim * 2
        self.score_head = nn.Sequential(
            nn.Linear(pair_dim, score_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(score_hidden_dim, score_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(score_hidden_dim, 1),
        )

    def encode_segment(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        h_obs = self.state_encoder(obs)
        h_act = self.action_encoder(act)
        h = torch.cat([h_obs, h_act], dim=-1)
        return h

    def score_segment(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        h = self.encode_segment(obs, act)
        score = self.score_head(h).squeeze(-1)
        return score

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        s1 = self.score_segment(batch["x1_obs"], batch["x1_act"])
        s2 = self.score_segment(batch["x2_obs"], batch["x2_act"])
        logit = s1 - s2
        prob = torch.sigmoid(logit)

        return {
            "score_1": s1,
            "score_2": s2,
            "logit": logit,
            "prob": prob,
        }


@dataclass
class TrainConfig:
    dataset_path: str
    output_dir: str
    seed: int = 42
    batch_size: int = 64
    num_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    hidden_dim: int = 128
    num_tcn_blocks: int = 4
    kernel_size: int = 5
    dropout: float = 0.1
    score_hidden_dim: int = 128
    normalize_obs: bool = True
    normalize_act: bool = True
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def binary_accuracy_from_logits(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    preds = (logits >= 0).long()
    return (preds == targets).float().mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        targets = batch["y"].float()

        outputs = model(batch)
        logits = outputs["logit"]

        loss = F.binary_cross_entropy_with_logits(logits, targets)
        acc = binary_accuracy_from_logits(logits, batch["y"])

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bsz = batch["y"].shape[0]
        total_loss += float(loss.item()) * bsz
        total_acc += float(acc.item()) * bsz
        total_n += bsz

    return {
        "loss": total_loss / max(total_n, 1),
        "acc": total_acc / max(total_n, 1),
    }


def build_loaders(
    config: TrainConfig,
) -> Tuple[DataLoader, DataLoader, HopperPairDataset]:
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


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train BT baseline on Hopper pairwise preference data."
    )
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
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-tcn-blocks", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--score-hidden-dim", type=int, default=128)
    parser.add_argument("--normalize-obs", action="store_true")
    parser.add_argument("--normalize-act", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

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
        score_hidden_dim=args.score_hidden_dim,
        normalize_obs=args.normalize_obs,
        normalize_act=args.normalize_act,
        num_workers=args.num_workers,
        device=args.device,
    )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, dataset = build_loaders(config)
    device = torch.device(config.device)

    model = PairwiseBTModel(
        obs_dim=dataset.obs_dim,
        act_dim=dataset.act_dim,
        hidden_dim=config.hidden_dim,
        num_tcn_blocks=config.num_tcn_blocks,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        score_hidden_dim=config.score_hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    config_dict = asdict(config)
    config_dict["dataset_stats"] = dataset.get_stats()
    save_json(output_dir / "config.json", config_dict)

    best_val_acc = -math.inf
    history = []

    print("=== DATASET STATS ===")
    print(json.dumps(dataset.get_stats(), indent=2))
    print("\n=== TRAINING ===")

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        val_metrics = run_epoch(model, val_loader, None, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        history.append(row)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.4f} | "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_acc']:.4f}"
        )

        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config_dict,
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                },
                output_dir / "best_model.pt",
            )

    save_json(output_dir / "history.json", {"history": history})
    print(f"\nBest val acc: {best_val_acc:.4f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
