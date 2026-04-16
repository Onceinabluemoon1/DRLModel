import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="name=path_to_infomec_metrics.json")
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for item in args.inputs:
        name, path = item.split("=", 1)
        with open(path, "r") as f:
            data = json.load(f)
        rows.append({
            "name": name,
            "infom": data.get("infom"),
            "infoc": data.get("infoc"),
            "infoe": data.get("infoe"),
            "active_latents": sum(bool(x) for x in data.get("active_latents", [])),
            "latent_dim": data.get("latent_dim"),
            "first_probe_acc": (data.get("linear_probe_accuracy_per_source") or [None])[0],
        })

    csv_path = os.path.join(args.outdir, "infomec_summary.csv")
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
        plt.savefig(os.path.join(args.outdir, f"{metric}_across_runs.png"), dpi=200)
        plt.close()

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
