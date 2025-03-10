"""
Visualization module for QA Dataset Clustering.

This module provides functions for visualizing clustering results and evaluation
metrics.
"""

import os
from collections import Counter
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from qadst.logging import get_logger

logger = get_logger(__name__)


def plot_cluster_distribution(
    cluster_assignments: List[int],
    title: str,
    color: str,
    log_scale: bool = False,
    ax=None,
) -> None:
    """
    Plot the distribution of cluster sizes.

    Args:
        cluster_assignments: List of cluster assignments
        title: Title for the plot
        color: Color for the bars
        log_scale: Whether to use log-log scale for the plot
        ax: Matplotlib axes object to plot on (if None, uses current axes)
    """
    # Use provided axes or get current axes
    if ax is None:
        ax = plt.gca()

    if log_scale:
        # For log-log plot, we count how many clusters have each size
        cluster_counts = Counter(cluster_assignments)

        # Count frequency of each cluster size
        size_frequency = Counter(cluster_counts.values())

        # Convert to lists for plotting
        sizes = sorted(size_frequency.keys())  # Unique cluster sizes
        frequencies = [
            size_frequency[size] for size in sizes
        ]  # Number of clusters with each size

        # Filter out zeros for log scale
        valid_indices = [
            i for i, freq in enumerate(frequencies) if freq > 0 and sizes[i] > 0
        ]
        valid_sizes = [sizes[i] for i in valid_indices]
        valid_frequencies = [frequencies[i] for i in valid_indices]

        # Use loglog for proper log-log plotting
        ax.loglog(
            valid_sizes,
            valid_frequencies,
            marker="o",
            linestyle="-",
            color=color,
            alpha=0.8,
        )
        title += " (Log-Log Scale)"

        # Set appropriate labels for this distribution view
        ax.set_xlabel("Cluster Size (number of items, log scale)")
        ax.set_ylabel("Number of Clusters of Size X (log scale)")
    else:
        # For linear scale, keep the original rank-based visualization
        cluster_counts = Counter(cluster_assignments)
        sizes = sorted(cluster_counts.values(), reverse=True)
        ranks = list(range(1, len(sizes) + 1))

        # Use bar plot for linear scale
        ax.bar(ranks, sizes, color=color, alpha=0.6)

        # For linear plots, use standard labels
        ax.set_xlabel("Cluster Rank (largest to smallest)")
        ax.set_ylabel("Number of Items in Cluster")

    ax.set_title(title)

    # Add grid for better readability, especially in log scale
    ax.grid(True, which="both", ls="-", alpha=0.2)


def visualize_silhouette_score(
    reports: Dict[str, Dict[str, Any]],
    output_dir: str,
    filename: str = "silhouette_comparison.png",
) -> str:
    """
    Visualize silhouette scores for different clustering models.

    Args:
        reports: Dictionary mapping model names to their evaluation reports
        output_dir: Directory to save the visualization
        filename: Name of the output file

    Returns:
        Path to the saved visualization file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Extract silhouette scores
    models = []
    scores = []

    for model_name, report in reports.items():
        models.append(model_name)
        scores.append(report["metrics"]["silhouette_score"])

    # Create the visualization
    plt.figure(figsize=(10, 6))
    plt.bar(models, scores, color=["blue", "red"])
    plt.title("Silhouette Score Comparison")
    plt.xlabel("Clustering Model")
    plt.ylabel("Silhouette Score")
    plt.ylim(-1, 1)  # Silhouette score range

    # Add value labels on top of bars
    for i, score in enumerate(scores):
        plt.text(i, score + 0.05, f"{score:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Silhouette score visualization saved to {output_path}")
    return output_path


def plot_cluster_distributions(
    dp_clusters: List[int],
    pyp_clusters: List[int],
    output_dir: str,
    plot_type: str = "linear",
) -> str:
    """
    Generate plots comparing cluster distributions for DP and PYP.

    Args:
        dp_clusters: List of cluster assignments from Dirichlet Process
        pyp_clusters: List of cluster assignments from Pitman-Yor Process
        output_dir: Directory to save the plots
        plot_type: Type of plot to generate ("linear" or "log-log")

    Returns:
        Path to the saved plot file
    """
    os.makedirs(output_dir, exist_ok=True)

    log_scale = plot_type == "log-log"
    filename = (
        "cluster_distribution_log.png" if log_scale else "cluster_distribution.png"
    )
    output_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(12, 5))

    # Plot DP clusters
    plt.subplot(1, 2, 1)
    plot_cluster_distribution(
        dp_clusters,
        "Dirichlet Process Cluster Sizes",
        "blue",
        log_scale=log_scale,
    )

    # Plot PYP clusters
    plt.subplot(1, 2, 2)
    plot_cluster_distribution(
        pyp_clusters,
        "Pitman-Yor Process Cluster Sizes",
        "red",
        log_scale=log_scale,
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    log_msg = "Log-log scale " if log_scale else ""
    logger.info(f"{log_msg}Cluster distribution plot saved to {output_path}")

    return output_path
