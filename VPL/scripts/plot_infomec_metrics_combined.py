#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description='Plot InfoM/InfoE/InfoC together from an InfoMEC sweep directory.')
    p.add_argument('--runs-root', required=True, help='Directory containing beta*/infomec_metrics.json')
    p.add_argument('--out', required=True, help='Output PNG path')
    p.add_argument('--title', default='InfoMEC metrics across runs')
    return p.parse_args()


def beta_from_name(name: str) -> float:
    m = re.search(r'beta([0-9p\.]+)', name)
    if not m:
        raise ValueError(f'Could not parse beta from run name: {name}')
    return float(m.group(1).replace('p', '.'))


def main():
    args = parse_args()
    root = Path(args.runs_root)
    rows = []
    for run_dir in root.iterdir():
        metrics_path = run_dir / 'infomec_metrics.json'
        if run_dir.is_dir() and metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            rows.append({
                'name': run_dir.name,
                'beta': beta_from_name(run_dir.name),
                'infom': metrics.get('infom'),
                'infoe': metrics.get('infoe'),
                'infoc': metrics.get('infoc'),
            })
    rows.sort(key=lambda r: r['beta'])
    betas = [r['beta'] for r in rows]
    beta_labels = [str(r['beta']) for r in rows]
    x = list(range(len(rows)))

    plt.figure(figsize=(7.8, 4.2))
    plt.plot(x, [r['infom'] for r in rows], marker='o', label='InfoM')
    plt.plot(x, [r['infoe'] for r in rows], marker='o', label='InfoE')
    plt.plot(x, [r['infoc'] for r in rows], marker='o', label='InfoC')
    plt.xlabel('beta')
    plt.ylabel('score')
    plt.ylim(0, 1.05)
    plt.xlim(-0.2, len(x) - 0.8)
    plt.xticks(x, beta_labels)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title(args.title)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
