"""
Visualization module for QA Dataset Clustering.

This module provides functions for visualizing clustering results and evaluation
metrics.
"""

import os
from collections import Counter
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from qadst.logging import get_logger

logger = get_logger(__name__)

# Global color scheme for consistent visualization
MODEL_COLORS = {"Dirichlet": "blue", "Pitman-Yor": "orange"}


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
    output_path = os.path.join(output_dir, filename)

    # Extract silhouette scores
    models = []
    scores = []
    colors = []

    for model_name, report in reports.items():
        models.append(model_name)
        scores.append(report["silhouette_score"])
        colors.append(MODEL_COLORS.get(model_name, "gray"))

    # Create the visualization
    plt.figure(figsize=(10, 6))
    plt.bar(models, scores, color=colors)
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


def _plot_cluster_size_distribution(reports, ax):
    """
    Plot cluster size distributions for each model.

    Args:
        reports: Dictionary mapping model names to their evaluation reports
        ax: Matplotlib axes object to plot on
    """
    for model_name, report in reports.items():
        # Check if we have the required data
        has_metrics = "basic_metrics" in report
        has_dist_key = "cluster_size_distribution"
        has_distribution = has_metrics and has_dist_key in report["basic_metrics"]

        if not has_distribution:
            logger.warning(f"Skipping {model_name}: No cluster size distribution data")
            continue

        # Use pre-computed cluster size distribution from basic_metrics
        cluster_size_dist = report["basic_metrics"][has_dist_key]

        # Convert string keys to integers and create a Counter
        size_frequency = Counter()
        for cluster_id, size in cluster_size_dist.items():
            size_frequency[size] += 1

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

        color = MODEL_COLORS.get(model_name, "gray")

        # Examples of powerlaw_params:
        #
        #   {'alpha': 1.2941766512739343, 'xmin': 1.0, 'is_powerlaw': True}
        #   {'alpha': 2.8547451299978364, 'xmin': 4.0, 'is_powerlaw': True}
        #
        powerlaw_params = report.get("powerlaw_params", {})

        alpha = powerlaw_params.get("alpha", None)
        alpha_str = f"α={alpha:.2f}," if alpha is not None else ""

        is_powerlaw = powerlaw_params.get("is_powerlaw", False)
        status = " follows power-law" if is_powerlaw else " non power-law"

        label_details = f"{alpha_str}{status}".strip(", ")
        label = f"{model_name} ({label_details})"

        # Plot rank vs size
        ax.loglog(
            valid_sizes,
            valid_frequencies,
            marker="o",
            linestyle="--",
            label=label,
            color=color,
            alpha=0.7,
        )

    ax.set_title("Cluster Size Distribution (Log-Log Scale)")
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Number of Clusters")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()


def _plot_silhouette_scores(reports, ax):
    """
    Plot silhouette scores for each model.

    Args:
        reports: Dictionary mapping model names to their evaluation reports
        ax: Matplotlib axes object to plot on
    """
    # Extract silhouette scores
    models = []
    scores = []
    colors = []

    for model_name, report in reports.items():
        if "silhouette_score" in report:
            models.append(model_name)
            scores.append(report["silhouette_score"])
            colors.append(MODEL_COLORS.get(model_name, "gray"))

    # Create bar chart
    ax.bar(models, scores, color=colors)
    ax.set_title("Silhouette Score Comparison")
    ax.set_xlabel("Clustering Model")
    ax.set_ylabel("Silhouette Score")
    ax.set_ylim(-1, 1)  # Silhouette score range

    # Add value labels on top of bars
    for i, score in enumerate(scores):
        ax.text(i, score + 0.05, f"{score:.4f}", ha="center")


def _plot_similarity_metrics(reports, ax):
    """
    Plot similarity metrics for each model.

    Args:
        reports: Dictionary mapping model names to their evaluation reports
        ax: Matplotlib axes object to plot on
    """
    has_similarity_data = False

    # Track models for legend
    model_data = {}

    # Prepare data for grouped bar chart
    for i, (model_name, report) in enumerate(reports.items()):
        if "similarity_metrics" in report:
            has_similarity_data = True
            color = MODEL_COLORS.get(model_name, "gray")

            # Get metrics
            intra = report["similarity_metrics"].get("intra_cluster_mean", 0)
            inter = report["similarity_metrics"].get("inter_cluster_mean", 0)

            # Store for plotting
            model_data[model_name] = {
                "intra": intra,
                "inter": inter,
                "color": color,
                "position": i,
            }

    if has_similarity_data:
        # Set up positions
        x = np.arange(2)  # Two groups: intra and inter
        width = 0.35

        # Plot bars for each model
        for model_name, data in model_data.items():
            offset = data["position"] * width - (len(model_data) - 1) * width / 2
            ax.bar(
                x + offset,
                [data["intra"], data["inter"]],
                width,
                label=model_name,
                color=data["color"],
            )

        # Set labels
        ax.set_xticks(x)
        ax.set_xticklabels(["Intra-cluster", "Inter-cluster"])
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "Similarity metrics not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title("Similarity Comparison")
    ax.set_ylabel("Cosine Similarity")


def _plot_cluster_counts(reports, ax):
    """
    Plot number of clusters for each model.

    Args:
        reports: Dictionary mapping model names to their evaluation reports
        ax: Matplotlib axes object to plot on
    """
    # Extract number of clusters
    models = []
    num_clusters = []
    colors = []

    for model_name, report in reports.items():
        if "basic_metrics" in report and "num_clusters" in report["basic_metrics"]:
            models.append(model_name)
            num_clusters.append(report["basic_metrics"]["num_clusters"])
            colors.append(MODEL_COLORS.get(model_name, "gray"))

    # Create bar chart
    ax.bar(models, num_clusters, color=colors)
    ax.set_title("Number of Clusters")
    ax.set_xlabel("Clustering Model")
    ax.set_ylabel("Count")

    # Add value labels on top of bars
    for i, count in enumerate(num_clusters):
        ax.text(i, count + 0.5, str(count), ha="center")


def _plot_powerlaw_fit(reports, ax):
    """
    Plot power-law fit for cluster size distributions.

    Args:
        reports: Dictionary mapping model names to their evaluation reports
        ax: Matplotlib axes object to plot on
    """
    try:
        import powerlaw  # type: ignore

        # Examples of powerlaw_params:
        #
        #   {'alpha': 1.2941766512739343, 'xmin': 1.0, 'is_powerlaw': True}
        #   {'alpha': 2.8547451299978364, 'xmin': 4.0, 'is_powerlaw': True}
        #
        for model_name, report in reports.items():
            if "powerlaw_params" not in report or not report["powerlaw_params"].get(
                "alpha"
            ):
                continue

            # Check if we have the required data
            has_metrics = "basic_metrics" in report
            has_dist_key = "cluster_size_distribution"
            has_distribution = has_metrics and has_dist_key in report["basic_metrics"]

            if not has_distribution:
                logger.warning(
                    f"Skipping {model_name}: No cluster size distribution data"
                )
                continue

            # Get cluster sizes from the distribution
            cluster_dist = report["basic_metrics"]["cluster_size_distribution"]
            sizes = list(cluster_dist.values())

            if not sizes:
                continue

            # Get color for this model
            color = MODEL_COLORS.get(model_name, "gray")

            # Get power-law status
            is_powerlaw = report["powerlaw_params"].get("is_powerlaw", False)
            status = "follows power-law" if is_powerlaw else "non power-law"

            fit = powerlaw.Fit(sizes, discrete=True)

            # Plot the empirical PDF with the model's color
            fit.plot_pdf(ax=ax, color=color, linewidth=2, label=f"{model_name} (data)")

            # Plot the power-law fit with the same color but dashed line
            fit.power_law.plot_pdf(
                ax=ax,
                color=color,
                linestyle="--",
                label=f"{model_name} (α={fit.alpha:.2f}, {status})",
            )

        ax.set_title("Power-law Fit")
        ax.set_xlabel("Cluster Size")
        ax.set_ylabel("Probability Density")
        ax.legend(loc="lower left", fontsize=8)

    except ImportError:
        ax.text(
            0.5,
            0.5,
            "powerlaw package not installed",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        logger.warning(
            "powerlaw package not installed. Cannot visualize power-law fit."
        )


def _plot_outliers(reports, ax):
    """
    Plot outlier score distributions.

    Args:
        reports: Dictionary mapping model names to their evaluation reports
        ax: Matplotlib axes object to plot on
    """
    for model_name, report in reports.items():
        if "outliers" not in report or not report["outliers"]:
            continue

        # Get outlier scores
        outlier_scores = list(report["outliers"].values())
        if not outlier_scores:
            continue

        # Plot histogram of outlier scores
        color = MODEL_COLORS.get(model_name, "gray")
        ax.hist(outlier_scores, bins=20, alpha=0.7, color=color, label=model_name)

    ax.set_title("Outlier Score Distribution")
    ax.set_xlabel("Outlier Score")
    ax.set_ylabel("Frequency")
    ax.legend()


def visualize_evaluation_dashboard(
    reports: Dict[str, Dict[str, Any]],
    output_dir: str,
    filename: str = "evaluation_dashboard.png",
) -> str:
    """
    Generate a comprehensive dashboard visualization of evaluation metrics.

    This creates a 3x2 grid of plots showing:
    1. Cluster size distribution (log-log scale)
    2. Silhouette score comparison
    3. Similarity metrics comparison
    4. Power-law fit visualization
    5. Outlier distribution
    6. Number of clusters comparison

    Args:
        reports: Dictionary mapping model names to their evaluation reports
        output_dir: Directory to save the visualization
        filename: Name of the output file

    Returns:
        Path to the saved visualization file
    """
    output_path = os.path.join(output_dir, filename)

    # Create figure with 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    # 1. Cluster size distribution (top-left)
    _plot_cluster_size_distribution(reports, axes[0, 0])

    # 2. Silhouette score comparison (top-right)
    _plot_silhouette_scores(reports, axes[0, 1])

    # 3. Similarity metrics (middle-left)
    _plot_similarity_metrics(reports, axes[1, 0])

    # 4. Power-law fit (middle-right)
    _plot_powerlaw_fit(reports, axes[1, 1])

    # 5. Outlier distribution (bottom-left)
    _plot_outliers(reports, axes[2, 0])

    # 6. Number of clusters (bottom-right)
    _plot_cluster_counts(reports, axes[2, 1])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Evaluation dashboard saved to {output_path}")
    return output_path
