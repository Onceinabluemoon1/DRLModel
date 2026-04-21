import argparse
import json
import os
import matplotlib.pyplot as plt

def parse_beta(name: str) -> float:
    s = name.replace("beta", "").replace("p", ".")
    return float(s)

def load_metric(scan_root, beta_name):
    path = os.path.join(scan_root, beta_name, "metrics.json")
    with open(path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Epoch-450 beta tradeoffs")
    args = parser.parse_args()

    beta_dirs = sorted(
        [d for d in os.listdir(args.scan_root) if d.startswith("beta")],
        key=parse_beta,
    )

    betas, recon, kld, acc = [], [], [], []
    for d in beta_dirs:
        m = load_metric(args.scan_root, d)
        betas.append(parse_beta(d))
        recon.append(m["reconstruction_loss"])
        kld.append(m["kld_loss"])
        acc.append(m["accuracy"])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(betas, recon, marker="o", label="Reconstruction loss")
    ax1.plot(betas, kld, marker="o", label="KL loss")
    ax2.plot(betas, acc, marker="o", label="Accuracy")

    ax1.set_xlabel("beta")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(args.title)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

if __name__ == "__main__":
    main()