import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import wandb


def plot_embeddings(
    embeddings,
    labels=None,
    method="umap",
    n_components=2,
    title=None,
    filename=None,
    colormap="tab20",
    marker_size=5,
    alpha=0.7,
    figsize=(12, 10),
    legend=True,
    highlight_indices=None,
    highlight_color="red",
    highlight_marker="x",
    highlight_size=20,
    highlight_label="Highlighted",
):
    """
    Plot embeddings using dimensionality reduction.

    Args:
        embeddings: Tensor or array of embeddings
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ('umap', 'tsne', or 'pca')
        n_components: Number of dimensions to reduce to (2 or 3)
        title: Plot title
        filename: If provided, save plot to this file
        colormap: Matplotlib colormap for labels
        marker_size: Size of markers
        alpha: Transparency of markers
        figsize: Figure size
        legend: Whether to show legend
        highlight_indices: Optional indices to highlight with different color/marker
        highlight_color: Color for highlighted points
        highlight_marker: Marker style for highlighted points
        highlight_size: Size for highlighted points
        highlight_label: Label for highlighted points in legend

    Returns:
        Figure object
    """
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Apply dimensionality reduction
    if method.lower() == "umap":
        reducer = UMAP(n_components=n_components, random_state=42)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method.lower() == "pca":
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from 'umap', 'tsne', or 'pca'."
        )

    reduced_data = reducer.fit_transform(embeddings)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # 2D or 3D plot
    if n_components == 2:
        ax = fig.add_subplot(111)
    else:  # 3D
        ax = fig.add_subplot(111, projection="3d")

    # Create highlight mask if needed
    highlight_mask = np.zeros(len(embeddings), dtype=bool)
    if highlight_indices is not None:
        highlight_mask[highlight_indices] = True

    # Plot non-highlighted points
    if labels is not None:
        # Get unique labels for coloring
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap(colormap, len(unique_labels))

        # Plot each label with different color
        for i, label in enumerate(unique_labels):
            mask = (labels == label) & ~highlight_mask
            if n_components == 2:
                ax.scatter(
                    reduced_data[mask, 0],
                    reduced_data[mask, 1],
                    c=[cmap(i)],
                    s=marker_size,
                    alpha=alpha,
                    label=f"Class {label}",
                )
            else:  # 3D
                ax.scatter(
                    reduced_data[mask, 0],
                    reduced_data[mask, 1],
                    reduced_data[mask, 2],
                    c=[cmap(i)],
                    s=marker_size,
                    alpha=alpha,
                    label=f"Class {label}",
                )
    else:
        # Plot all points with same color
        mask = ~highlight_mask
        if n_components == 2:
            ax.scatter(
                reduced_data[mask, 0],
                reduced_data[mask, 1],
                s=marker_size,
                alpha=alpha,
                c="blue",
            )
        else:  # 3D
            ax.scatter(
                reduced_data[mask, 0],
                reduced_data[mask, 1],
                reduced_data[mask, 2],
                s=marker_size,
                alpha=alpha,
                c="blue",
            )

    # Plot highlighted points
    if highlight_indices is not None and np.any(highlight_mask):
        if n_components == 2:
            ax.scatter(
                reduced_data[highlight_mask, 0],
                reduced_data[highlight_mask, 1],
                c=highlight_color,
                s=highlight_size,
                marker=highlight_marker,
                label=highlight_label,
            )
        else:  # 3D
            ax.scatter(
                reduced_data[highlight_mask, 0],
                reduced_data[highlight_mask, 1],
                reduced_data[highlight_mask, 2],
                c=highlight_color,
                s=highlight_size,
                marker=highlight_marker,
                label=highlight_label,
            )

    # Add labels and legend
    if title:
        ax.set_title(title)

    if n_components == 2:
        ax.set_xlabel(f"{method.upper()} Dimension 1")
        ax.set_ylabel(f"{method.upper()} Dimension 2")
    else:  # 3D
        ax.set_xlabel(f"{method.upper()} Dimension 1")
        ax.set_ylabel(f"{method.upper()} Dimension 2")
        ax.set_zlabel(f"{method.upper()} Dimension 3")

    if legend:
        ax.legend(loc="best")

    plt.tight_layout()

    # Save figure if filename provided
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    return fig


def plot_cluster_distribution(
    cluster_labels, class_labels=None, title=None, filename=None
):
    """
    Plot the distribution of clusters, optionally with class information.

    Args:
        cluster_labels: Array of cluster assignments
        class_labels: Optional array of class labels
        title: Plot title
        filename: If provided, save plot to this file

    Returns:
        Figure object
    """
    # Count clusters
    unique_clusters = np.unique(cluster_labels)
    cluster_counts = np.bincount(cluster_labels)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot cluster distribution
    ax.bar(range(len(unique_clusters)), cluster_counts, alpha=0.7)

    # Add class distribution within clusters if provided
    if class_labels is not None:
        # Get unique classes
        unique_classes = np.unique(class_labels)
        cmap = plt.get_cmap("tab20", len(unique_classes))

        # For each cluster, plot class distribution
        bottom = np.zeros(len(unique_clusters))
        for i, class_idx in enumerate(unique_classes):
            # Count instances of this class in each cluster
            class_in_cluster = np.zeros(len(unique_clusters))
            for j, cluster_idx in enumerate(unique_clusters):
                mask = (cluster_labels == cluster_idx) & (class_labels == class_idx)
                class_in_cluster[j] = np.sum(mask)

            # Plot stacked bar
            ax.bar(
                range(len(unique_clusters)),
                class_in_cluster,
                bottom=bottom,
                alpha=0.7,
                label=f"Class {class_idx}",
                color=cmap(i),
            )

            # Update bottom for next class
            bottom += class_in_cluster

    # Add labels and legend
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    ax.set_xticks(range(len(unique_clusters)))
    ax.set_xticklabels([f"Cluster {i}" for i in unique_clusters])

    if class_labels is not None:
        ax.legend(loc="best")

    plt.tight_layout()

    # Save figure if filename provided
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    return fig


def plot_class_distribution(labels, title=None, filename=None, max_classes=20):
    """
    Plot the distribution of classes.

    Args:
        labels: Array of class labels
        title: Plot title
        filename: If provided, save plot to this file
        max_classes: Maximum number of classes to show individually (others grouped)

    Returns:
        Figure object
    """
    # Count classes
    unique_classes = np.unique(labels)
    class_counts = np.bincount(labels)

    # Sort by count (descending)
    sorted_idx = np.argsort(-class_counts)
    sorted_classes = unique_classes[sorted_idx]
    sorted_counts = class_counts[sorted_idx]

    # Group small classes if too many
    if len(unique_classes) > max_classes:
        main_classes = sorted_classes[: max_classes - 1]
        main_counts = sorted_counts[: max_classes - 1]
        other_count = np.sum(sorted_counts[max_classes - 1 :])

        plot_classes = np.append(main_classes, [-1])  # -1 for "Other"
        plot_counts = np.append(main_counts, [other_count])
    else:
        plot_classes = sorted_classes
        plot_counts = sorted_counts

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot class distribution
    bars = ax.bar(range(len(plot_classes)), plot_counts, alpha=0.7)

    # Color bars
    cmap = plt.get_cmap("viridis", len(plot_classes))
    for i, bar in enumerate(bars):
        bar.set_color(cmap(i))

    # Add labels
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)

    # Set x-tick labels
    ax.set_xticks(range(len(plot_classes)))
    labels = [f"Class {c}" if c != -1 else "Other" for c in plot_classes]
    ax.set_xticklabels(labels, rotation=90 if len(plot_classes) > 10 else 0)

    plt.tight_layout()

    # Save figure if filename provided
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    return fig


def log_embedding_plot_to_wandb(
    embeddings, labels=None, method="umap", title=None, highlight_indices=None
):
    """
    Create an embedding plot and log it to wandb.

    Args:
        embeddings: Tensor or array of embeddings
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ('umap', 'tsne', or 'pca')
        title: Plot title
        highlight_indices: Optional indices to highlight
    """
    if wandb.run is None:
        print(
            "Warning: wandb run not initialized, skipping log_embedding_plot_to_wandb"
        )
        return

    fig = plot_embeddings(
        embeddings=embeddings,
        labels=labels,
        method=method,
        title=title,
        highlight_indices=highlight_indices,
    )

    # Log to wandb
    wandb.log({title: wandb.Image(fig)})

    # Close figure to free memory
    plt.close(fig)
