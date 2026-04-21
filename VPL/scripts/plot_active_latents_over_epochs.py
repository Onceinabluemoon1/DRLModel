#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot number of active latents across training epochs.")
    p.add_argument("--runs-root", required=True,
                   help="Directory like .../infomec_over_training containing beta*/epoch_*/infomec_metrics.json")
    p.add_argument("--outdir", required=True,
                   help="Directory where plots will be written")
    p.add_argument("--title-prefix", default="Active latents over training")
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


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    beta_dirs = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("beta")]
    beta_dirs.sort(key=lambda p: beta_from_name(p.name))

    for beta_dir in beta_dirs:
        rows = []
        for epoch_dir in beta_dir.iterdir():
            metrics_path = epoch_dir / "infomec_metrics.json"
            if epoch_dir.is_dir() and metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                active = metrics.get("active_latents", [])
                rows.append({
                    "epoch": epoch_from_name(epoch_dir.name),
                    "num_active": int(sum(bool(x) for x in active)),
                    "latent_dim": len(active),
                })

        rows.sort(key=lambda r: r["epoch"])
        if not rows:
            print(f"Skipping {beta_dir.name}: no metrics found")
            continue

        epochs = [r["epoch"] for r in rows]
        counts = [r["num_active"] for r in rows]
        latent_dim = rows[0]["latent_dim"]
        beta = beta_from_name(beta_dir.name)

        plt.figure(figsize=(7.4, 4.2))
        plt.plot(epochs, counts, marker="o")
        plt.xlabel("epoch")
        plt.ylabel("number of active latents")
        plt.ylim(0, latent_dim + 0.5)
        plt.xlim(min(epochs), max(epochs))
        plt.grid(alpha=0.3)
        plt.title(f"{args.title_prefix} (beta={beta}, latent_dim={latent_dim})")
        plt.tight_layout()

        out_path = outdir / f"{beta_dir.name}_active_latents_over_epochs.png"
        plt.savefig(out_path, dpi=220)
        plt.close()
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()