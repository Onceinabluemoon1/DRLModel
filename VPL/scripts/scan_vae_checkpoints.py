#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
from pathlib import Path

import numpy as np
import torch
import gym
import jaxrl_m.envs  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pref_learn.models.utils import get_datasets


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate saved VAE checkpoints on eval loss / recon / KL / accuracy.')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--env', required=True)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--set-length', type=int, default=-1)
    p.add_argument('--device', default='auto')
    p.add_argument('--outdir', required=True)
    return p.parse_args()


def eval_ckpt(model, test_loader, device):
    select_losses = []
    weighted = {'loss': 0.0, 'reconstruction_loss': 0.0, 'kld_loss': 0.0, 'accuracy': 0.0}
    total_bs = 0
    with torch.no_grad():
        for batch in test_loader:
            obs1 = batch['observations'].to(device).float()
            obs2 = batch['observations_2'].to(device).float()
            labels = batch['labels'].to(device).float()
            _, metrics = model(obs1, obs2, labels)
            bs = obs1.shape[0]
            total_bs += bs
            select_losses.append(metrics['loss'])
            for k in weighted:
                weighted[k] += metrics[k] * bs
    for k in weighted:
        weighted[k] /= total_bs
    weighted['selection_score'] = float(np.mean(select_losses))
    return weighted


def plot_metric(rows, key, out):
    xs = [r['epoch'] for r in rows]
    ys = [r[key] for r in rows]
    plt.figure(figsize=(6.4, 4.0))
    plt.plot(xs, ys, marker='o')
    plt.xlabel('epoch')
    plt.ylabel(key)
    plt.title(f'{key} across checkpoints')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def main():
    args = parse_args()
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device

    gym_env = gym.make(args.env)
    observation_dim = (
        gym_env.reward_observation_space.shape[0]
        if hasattr(gym_env, 'reward_observation_space')
        else gym_env.observation_space.shape[0]
    )
    action_dim = gym_env.action_space.shape[0]

    _, test_loader, _, _, _, _, _ = get_datasets(
        args.dataset, observation_dim, action_dim, args.batch_size, args.set_length
    )

    run_dir = Path(args.run_dir)
    ckpts = sorted(
        glob.glob(str(run_dir / 'model_*.pt')),
        key=lambda p: int(re.search(r'model_(\d+)\.pt', os.path.basename(p)).group(1))
    )
    best_path = run_dir / 'best_model.pt'
    if best_path.exists():
        ckpts.append(str(best_path))

    rows = []
    for path in ckpts:
        model = torch.load(path, map_location=device, weights_only=False)
        model = model.to(device)
        model.eval()
        stats = eval_ckpt(model, test_loader, device)
        epoch_match = re.search(r'model_(\d+)\.pt', os.path.basename(path))
        epoch = int(epoch_match.group(1)) if epoch_match else -1
        rows.append({'checkpoint': os.path.basename(path), 'epoch': epoch, **stats})

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / 'checkpoint_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_rows = [r for r in rows if r['epoch'] >= 0]
    for key in ['selection_score', 'loss', 'reconstruction_loss', 'kld_loss', 'accuracy']:
        plot_metric(plot_rows, key, outdir / f'{key}_across_checkpoints.png')

    print(f'Wrote {csv_path}')


if __name__ == '__main__':
    main()
