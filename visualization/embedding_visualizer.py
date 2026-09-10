"""
Embedding Visualizer module for latent representation projections using t-SNE and UMAP.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingVisualizer:
    """Projects high-dimensional feature embeddings onto 2D manifold embeddings."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.plots_dir = Path(config["outputs"]["plots_dir"])
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.genres = config["genres"]

    def plot_tsne(
        self, features: np.ndarray, labels: np.ndarray, title: str = "t-SNE Latent Feature Embedding", save_name: str = "tsne_embedding"
    ) -> plt.Figure:
        """Computes and renders 2D t-SNE scatter plot."""
        logger.info(f"Computing t-SNE reduction for {len(features)} samples...")
        perplexity = min(30, max(5, len(features) - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=self.config["project"]["seed"])
        features_2d = tsne.fit_transform(features)

        fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
        sns.scatterplot(
            x=features_2d[:, 0],
            y=features_2d[:, 1],
            hue=[self.genres[l] for l in labels],
            palette="tab10",
            style=[self.genres[l] for l in labels],
            s=60,
            alpha=0.85,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Folk Genre")
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{save_name}.png")
        return fig

    def plot_umap(
        self, features: np.ndarray, labels: np.ndarray, title: str = "UMAP Latent Feature Embedding", save_name: str = "umap_embedding"
    ) -> plt.Figure:
        """Computes and renders 2D UMAP scatter plot (with PCA fallback if umap-learn is absent)."""
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=self.config["project"]["seed"])
            features_2d = reducer.fit_transform(features)
            label_title = title
        except ImportError:
            logger.warning("umap-learn not installed. Falling back to PCA for 2D embedding visualizer.")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=self.config["project"]["seed"])
            features_2d = pca.fit_transform(features)
            label_title = f"PCA Projection (UMAP Fallback) - {title}"

        fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
        sns.scatterplot(
            x=features_2d[:, 0],
            y=features_2d[:, 1],
            hue=[self.genres[l] for l in labels],
            palette="Set2",
            style=[self.genres[l] for l in labels],
            s=60,
            alpha=0.85,
            ax=ax,
        )
        ax.set_title(label_title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Dimension 1", fontsize=11)
        ax.set_ylabel("Dimension 2", fontsize=11)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Folk Genre")
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{save_name}.png")
        return fig
