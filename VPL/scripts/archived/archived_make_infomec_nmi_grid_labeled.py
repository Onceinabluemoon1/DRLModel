#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Make a grid of InfoMEC NMI heatmaps with active-latent counts in titles."
    )
    p.add_argument(
        "--runs-root",
        required=True,
        help="Directory containing beta*/nmi.npy and infomec_metrics.json",
    )
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional ordered run names, e.g. beta0 beta0p25 ...",
    )
    p.add_argument(
        "--source-label",
        default=None,
        help="Override y-axis label for single-source plots",
    )
    p.add_argument("--title", default="InfoMEC NMI heatmaps across runs")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.runs_root)
    if args.names:
        names = args.names
    else:
        names = sorted(
            [p.name for p in root.iterdir() if p.is_dir() and (p / "nmi.npy").exists()]
        )
    if not names:
        raise SystemExit("No run folders with nmi.npy found.")

    rows = []
    for name in names:
        run_dir = root / name
        with open(run_dir / "infomec_metrics.json", "r") as f:
            metrics = json.load(f)
        rows.append(
            {
                "name": name,
                "nmi": np.load(run_dir / "nmi.npy"),
                "active_count": sum(bool(x) for x in metrics.get("active_latents", [])),
                "latent_dim": metrics.get("latent_dim", 0),
                "source_names": metrics.get("source_names", ["source"]),
            }
        )

    cols = min(4, len(rows))
    nrows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(
        nrows, cols, figsize=(4.4 * cols, 2.8 * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).reshape(nrows, cols)

    im = None
    for ax, row in zip(axes.flat, rows):
        arr = row["nmi"]
        global_max = max(float(row["nmi"].max()) for row in rows)
        if global_max <= 0:
            global_max = 1.0
        im = ax.imshow(arr, aspect="auto", vmin=0, vmax=global_max)
        metrics_path = run_dir / "infomec_metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        infom = float(metrics.get("infom", float("nan")))
        infoc = float(metrics.get("infoc", float("nan")))
        active = metrics.get("active_latents", [])
        num_active = int(sum(bool(x) for x in active))
        latent_dim = int(len(active))

        ax.set_title(
            f"{row['name']} ({num_active}/{latent_dim})\nM={infom:.2f}, C={infoc:.2f}"
        )
        ax.set_xticks(range(arr.shape[1]))
        ax.set_xticklabels(
            [f"z{i}" for i in range(arr.shape[1])], rotation=90, fontsize=8
        )
        ax.set_yticks(range(arr.shape[0]))
        if arr.shape[0] == 1:
            label = args.source_label or row["source_names"][0]
            ax.set_yticklabels([label], fontsize=8)
        else:
            ax.set_yticklabels(row["source_names"], fontsize=8)

    for ax in axes.flat[len(rows) :]:
        ax.axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, label="Normalized MI")
    fig.suptitle(args.title, y=1.02)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
