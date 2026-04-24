#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot InfoM/InfoE/InfoC across training epochs with one subplot per beta."
    )
    p.add_argument(
        "--runs-root",
        required=True,
        help="Directory like .../infomec_over_training containing beta*/epoch_*/infomec_metrics.json",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output PNG path",
    )
    p.add_argument(
        "--title",
        default="InfoMEC over training",
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


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)

    beta_dirs = [
        p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("beta")
    ]
    beta_dirs.sort(key=lambda p: beta_from_name(p.name))

    if not beta_dirs:
        raise SystemExit("No beta* directories found.")

    n = len(beta_dirs)
    ncols = 4 if n >= 4 else n
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.3 * ncols, 3.5 * nrows), sharey=True
    )
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = list(axes.flat)

    for ax, beta_dir in zip(axes, beta_dirs):
        rows = load_rows(beta_dir)
        if not rows:
            ax.set_visible(False)
            continue

        beta = beta_from_name(beta_dir.name)
        epochs = [r["epoch"] for r in rows]

        ax.plot(epochs, [r["infom"] for r in rows], marker="o", label="InfoM")
        ax.plot(epochs, [r["infoe"] for r in rows], marker="o", label="InfoE")
        ax.plot(epochs, [r["infoc"] for r in rows], marker="o", label="InfoC")

        ax.set_title(f"beta={beta}")
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    # y-label only on first column
    for i, ax in enumerate(axes[:n]):
        if i % ncols == 0:
            ax.set_ylabel("score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(args.title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
