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
        description="Plot InfoM/InfoE/InfoC across training epochs."
    )
    p.add_argument(
        "--runs-root",
        required=True,
        help="Directory like .../infomec_over_training containing beta*/epoch_*/infomec_metrics.json",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Directory where the epoch plots will be written",
    )
    p.add_argument(
        "--title-prefix",
        default="InfoMEC over training",
        help="Prefix for plot titles",
    )
    return p.parse_args()


def beta_from_name(name: str) -> float:
    m = re.search(r"beta([0-9p\.]+)", name)
    if not m:
        raise ValueError(f"Could not parse beta from run name: {name}")
    return float(m.group(1).replace("p", "."))


def epoch_from_name(name: str) -> int:
    m = re.search(r"epoch_(\d+)", name)
    if not m:
        raise ValueError(f"Could not parse epoch from folder name: {name}")
    return int(m.group(1))


def load_beta_rows(beta_dir: Path):
    rows = []
    for epoch_dir in beta_dir.iterdir():
        metrics_path = epoch_dir / "infomec_metrics.json"
        if epoch_dir.is_dir() and metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            rows.append(
                {
                    "epoch": epoch_from_name(epoch_dir.name),
                    "infom": metrics.get("infom"),
                    "infoe": metrics.get("infoe"),
                    "infoc": metrics.get("infoc"),
                }
            )

    rows.sort(key=lambda r: r["epoch"])
    return rows


def make_single_plot(beta_dir: Path, outdir: Path, title_prefix: str):
    rows = load_beta_rows(beta_dir)
    if not rows:
        print(f"Skipping {beta_dir.name}: no epoch_*/infomec_metrics.json found")
        return

    beta = beta_from_name(beta_dir.name)
    epochs = [r["epoch"] for r in rows]

    plt.figure(figsize=(7.6, 4.4))
    plt.plot(epochs, [r["infom"] for r in rows], marker="o", label="InfoM")
    plt.plot(epochs, [r["infoe"] for r in rows], marker="o", label="InfoE")
    plt.plot(epochs, [r["infoc"] for r in rows], marker="o", label="InfoC")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.ylim(0, 1.05)
    plt.xlim(min(epochs), max(epochs))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title(f"{title_prefix} (beta={beta})")
    plt.tight_layout()

    out_path = outdir / f"{beta_dir.name}_infomec_over_epochs.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Wrote {out_path}")


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
        make_single_plot(beta_dir, outdir, args.title_prefix)


if __name__ == "__main__":
    main()