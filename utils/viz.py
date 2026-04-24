import os
from typing import Optional

import numpy as np


def _run_tsne_and_save(
    samples: np.ndarray,
    annotator_indices: np.ndarray,
    out_png_path: str,
    seed: Optional[int],
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except Exception as err:
        print("t-SNE visualization skipped (missing dependencies).")
        print("Install with: pip install scikit-learn matplotlib")
        return

    n_samples = samples.shape[0]
    if n_samples < 3:
        print("t-SNE visualization skipped (requires at least 3 samples).")
        return

    # Choose a safe perplexity < n_samples
    max_safe_perp = max(2, n_samples // 3)
    perplexity = min(30, max_safe_perp)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate=200,
        verbose=0,
    )
    embedding_2d = tsne.fit_transform(samples)

    # Ensure parent directory exists for plot
    parent = os.path.dirname(out_png_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    # Plot and save
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=annotator_indices,
        cmap="tab10",
        s=10,
        alpha=0.9,
        edgecolors="none",
    )
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    cbar = plt.colorbar(scatter, ticks=sorted(np.unique(annotator_indices)))
    cbar.set_label("Annotator index")
    if out_png_path.endswith(".png") or out_png_path.endswith(".pdf"):
        plt.savefig(out_png_path, bbox_inches="tight", dpi=150)
    else:
        plt.savefig(out_png_path + ".png", bbox_inches="tight", dpi=150)
    plt.close()
