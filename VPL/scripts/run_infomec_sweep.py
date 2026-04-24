#!/usr/bin/env python3
"""Run InfoMEC over multiple checkpoints (e.g. a beta / KL sweep)."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Pairs like beta1=path/to/best_model.pt",
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--split", choices=["all", "train", "eval"], default="eval")
    p.add_argument("--source-keys", default="auto")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--set-length", type=int, default=-1)
    p.add_argument("--discrete-latents", action="store_true")
    p.add_argument("--num-bins", type=int, default=10)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    script_path = Path(__file__).resolve().parent / "run_infomec_analysis.py"
    rows = []

    for item in args.inputs:
        name, ckpt = item.split("=", 1)
        run_dir = Path(args.outdir) / sanitize(name)
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script_path),
            "--ckpt",
            ckpt,
            "--dataset",
            args.dataset,
            "--env",
            args.env,
            "--outdir",
            str(run_dir),
            "--split",
            args.split,
            "--source-keys",
            args.source_keys,
            "--batch-size",
            str(args.batch_size),
            "--set-length",
            str(args.set_length),
            "--num-bins",
            str(args.num_bins),
            "--device",
            args.device,
        ]
        if args.discrete_latents:
            cmd.append("--discrete-latents")

        subprocess.run(cmd, check=True)

        metrics_path = run_dir / "infomec_metrics.json"
        with open(metrics_path, "r") as f:
            data = json.load(f)

        rows.append(
            {
                "name": name,
                "ckpt": ckpt,
                "metrics_path": str(metrics_path),
                "infom": data.get("infom"),
                "infoc": data.get("infoc"),
                "infoe": data.get("infoe"),
                "active_latents": sum(bool(x) for x in data.get("active_latents", [])),
                "latent_dim": data.get("latent_dim"),
                "first_probe_acc": (
                    data.get("linear_probe_accuracy_per_source") or [None]
                )[0],
            }
        )

    csv_path = Path(args.outdir) / "infomec_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    names = [r["name"] for r in rows]
    for metric in ["infom", "infoc", "infoe", "first_probe_acc"]:
        vals = [r[metric] for r in rows]
        plt.figure(figsize=(max(6, len(names) * 0.7), 4))
        plt.plot(names, vals, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(f"{metric} across runs")
        plt.tight_layout()
        plt.savefig(Path(args.outdir) / f"{metric}_across_runs.png", dpi=200)
        plt.close()

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
