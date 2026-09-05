"""Shared preprocessing, paths, and plotting for the KBC examples."""

from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from examples.data import SPATIAL_DATASETS


EXAMPLES_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = EXAMPLES_DIR / "results"


def preprocess_features(
    key: str, features: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Apply dataset-specific feature preprocessing shared by train and predict."""
    if key != "dlpfc_151507":
        return features

    reduced = PCA(n_components=50, random_state=42).fit_transform(features)
    distances = cdist(positions, positions)
    neighbors = np.argsort(distances, axis=1)[:, 1:7]

    adjacency = np.zeros((len(positions), len(positions)))
    adjacency[np.arange(len(positions))[:, None], neighbors] = 1
    adjacency += np.eye(len(positions))
    adjacency /= adjacency.sum(axis=1, keepdims=True)

    propagated = [reduced]
    for _ in range(7):
        propagated.append(adjacency @ propagated[-1])
    return np.concatenate(propagated, axis=1)


def make_visualization_embedding(
    key: str, features: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Return deterministic two-dimensional coordinates for result plots."""
    plot_features = positions if key in SPATIAL_DATASETS else features
    if plot_features.shape[1] <= 2:
        return plot_features

    perplexity = min(30, len(plot_features) - 1)
    return TSNE(
        n_components=2, random_state=42, perplexity=perplexity
    ).fit_transform(plot_features)


def save_visualization(
    embedding: np.ndarray,
    y_true: np.ndarray,
    labels: np.ndarray,
    key: str,
    output_dir: str | Path,
) -> None:
    """Save reference and predicted-label plots beside the other run artifacts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def colors_for(count: int):
        if count <= 10:
            cmap = plt.get_cmap("tab10")
            return [cmap(i) for i in range(count)]
        if count <= 20:
            cmap = plt.get_cmap("tab20")
            return [cmap(i) for i in range(count)]
        return plt.cm.hsv(np.linspace(0, 1, count))

    plots = (
        (y_true, f"{key}_dataset.jpg"),
        (labels, f"{key}_clustering_result.jpg"),
    )
    for plot_labels, filename in plots:
        colors = colors_for(len(np.unique(plot_labels)))
        _, axis = plt.subplots(figsize=(8, 6))
        axis.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=plot_labels,
            cmap=ListedColormap(colors),
            alpha=0.7,
            s=15,
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Visualizations saved to {output_dir}")
