#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot InfoM/InfoE/InfoC and active latent count over training epochs."
    )
    p.add_argument(
        "--runs-root",
        required=True,
        help="Directory like .../infomec_over_training containing beta*/epoch_*/infomec_metrics.json",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Directory where the plots will be written",
    )
    p.add_argument(
        "--title-prefix",
        default="InfoMEC + active latents over training",
        help="Prefix for plot titles",
    )
    return p.parse_args()


def beta_from_name(name: str) -> float:
    m = re.search(r"beta([0-9p\.]+)", name)
    if not m:
        raise ValueError(f"Could not parse beta from {name}")
    return float(m.group(1).replace("p", "."))


def epoch_from_name(name: str) -> int:
    m = re.search(r"epoch_(\d+)", name)
    if not m:
        raise ValueError(f"Could not parse epoch from {name}")
    return int(m.group(1))


def load_rows(beta_dir: Path):
    rows = []
    for epoch_dir in beta_dir.iterdir():
        metrics_path = epoch_dir / "infomec_metrics.json"
        if epoch_dir.is_dir() and metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            active = metrics.get("active_latents", [])
            rows.append(
                {
                    "epoch": epoch_from_name(epoch_dir.name),
                    "infom": metrics.get("infom"),
                    "infoe": metrics.get("infoe"),
                    "infoc": metrics.get("infoc"),
                    "num_active": int(sum(bool(x) for x in active)),
                    "latent_dim": len(active),
                }
            )

    rows.sort(key=lambda r: r["epoch"])
    return rows


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    beta_dirs = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("beta")]
    beta_dirs.sort(key=lambda p: beta_from_name(p.name))

    if not beta_dirs:
        raise SystemExit("No beta* directories found.")

    for beta_dir in beta_dirs:
        rows = load_rows(beta_dir)
        if not rows:
            print(f"Skipping {beta_dir.name}: no epoch metrics found")
            continue

        beta = beta_from_name(beta_dir.name)
        latent_dim = rows[0]["latent_dim"]
        epochs = [r["epoch"] for r in rows]

        fig, ax1 = plt.subplots(figsize=(8.2, 4.8))

        l1 = ax1.plot(epochs, [r["infom"] for r in rows], marker="o", label="InfoM")
        l2 = ax1.plot(epochs, [r["infoe"] for r in rows], marker="o", label="InfoE")
        l3 = ax1.plot(epochs, [r["infoc"] for r in rows], marker="o", label="InfoC")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("InfoMEC score")
        ax1.set_ylim(0, 1.05)
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        l4 = ax2.plot(
            epochs,
            [r["num_active"] for r in rows],
            marker="s",
            linestyle="--",
            label="# active latents",
        )
        ax2.set_ylabel("number of active latents")
        ax2.set_ylim(0, latent_dim + 0.5)

        lines = l1 + l2 + l3 + l4
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="best")

        plt.title(f"{args.title_prefix} (beta={beta}, latent_dim={latent_dim})")
        plt.tight_layout()

        out_path = outdir / f"{beta_dir.name}_infomec_with_active_latents_over_epochs.png"
        plt.savefig(out_path, dpi=220)
        plt.close()
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()