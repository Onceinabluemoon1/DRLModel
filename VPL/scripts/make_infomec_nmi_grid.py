#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description='Make a grid of InfoMEC NMI heatmaps from beta run folders.')
    p.add_argument('--runs-root', required=True, help='Directory containing beta*/nmi.npy')
    p.add_argument('--out', required=True, help='Output PNG path')
    p.add_argument('--names', nargs='*', default=None, help='Optional ordered run names, e.g. beta0 beta0p25 ...')
    p.add_argument('--source-label', default='source_mode')
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.runs_root)
    if args.names:
        names = args.names
    else:
        names = sorted([p.name for p in root.iterdir() if p.is_dir() and (p / 'nmi.npy').exists()])
    if not names:
        raise SystemExit('No run folders with nmi.npy found.')

    nmis = [np.load(root / name / 'nmi.npy') for name in names]
    vmax = max(float(a.max()) for a in nmis) if nmis else 1.0
    n = len(names)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.2 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    im = None
    for ax, name, arr in zip(axes.flat, names, nmis):
        im = ax.imshow(arr, aspect='auto', vmin=0, vmax=vmax)
        ax.set_title(name)
        ax.set_yticks([0])
        ax.set_yticklabels([args.source_label])
        d = arr.shape[1]
        xt = list(range(0, d, max(1, d // 8)))
        if xt[-1] != d - 1:
            xt.append(d - 1)
        xt = sorted(set(xt))
        ax.set_xticks(xt)
        ax.set_xticklabels([f'z{i}' for i in xt], rotation=90, fontsize=8)
    for ax in axes.flat[len(names):]:
        ax.axis('off')
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label='Normalized MI')
    fig.suptitle('InfoMEC NMI heatmaps across runs', y=1.02)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220, bbox_inches='tight')
    print(f'Wrote {out}')

if __name__ == '__main__':
    main()
