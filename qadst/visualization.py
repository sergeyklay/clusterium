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
from matplotlib import colormaps

from qadst.logging import get_logger

logger = get_logger(__name__)

plt.style.use("default")


def get_model_colors(model_names):
    """Generate consistent colors for models using academically popular colormaps.

    Uses 'Set1' and 'tab10' colormaps which are standard in academic publications.
    """
    num_models = len(model_names)

    # For fewer models, use Set1 which has distinct colors commonly used in publications
    if num_models <= 9:
        cmap = colormaps["Set1"]
        colors = [cmap(i / max(1, 8)) for i in range(num_models)]
    # For more models, use tab10 which supports up to 10 distinct colors
    elif num_models <= 10:
        cmap = colormaps["tab10"]
        colors = [cmap(i / max(1, 9)) for i in range(num_models)]
    # For even more models, combine colors from both colormaps
    else:
        set1_cmap = colormaps["Set1"]
        tab10_cmap = colormaps["tab10"]
        colors = []
        for i in range(num_models):
            if i < 9:
                colors.append(set1_cmap(i / 8))
            else:
                colors.append(tab10_cmap((i - 9) / 9))

    return dict(zip(model_names, colors))


def _plot_cluster_size_distribution(reports, ax):
    """
    Plot cluster size distributions for each model.

    Args:
        reports: Dictionary mapping model names to their evaluation reports
        ax: Matplotlib axes object to plot on
    """
    # Generate colors for models
    model_colors = get_model_colors(list(reports.keys()))

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

        color = model_colors.get(model_name)

        # Plot rank vs size
        ax.loglog(
            valid_sizes,
            valid_frequencies,
            marker="o",
            linestyle="--",
            label=model_name,
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
    # Extract data from reports
    models, scores, error_models, zero_score_models = _extract_silhouette_data(reports)

    # Handle case where no valid scores are available
    if not models:
        _show_silhouette_message(ax, error_models, zero_score_models)
        return

    # Generate colors for models
    model_colors = get_model_colors(models)
    colors = [model_colors[model] for model in models]

    # Create bar chart
    ax.bar(models, scores, color=colors)
    ax.set_title("Silhouette Score Comparison")
    ax.set_xlabel("Clustering Model")
    ax.set_ylabel("Silhouette Score")
    ax.set_ylim(-1, 1)  # Silhouette score range

    # Add value labels on top of bars
    for i, score in enumerate(scores):
        ax.text(i, score + 0.05, f"{score:.4f}", ha="center")

    # If some models had errors or zero scores, add a note
    if error_models or zero_score_models:
        _add_silhouette_note(ax, error_models, zero_score_models)


def _extract_silhouette_data(reports):
    """Extract and categorize silhouette scores from reports."""
    models = []
    scores = []
    error_models = []
    zero_score_models = []

    for model_name, report in reports.items():
        if "silhouette_score" in report:
            score = report["silhouette_score"]
            if score == 0.0:
                # A score of exactly 0.0 often indicates calculation issues
                zero_score_models.append(model_name)
            elif score != -1:  # Check for special error value
                models.append(model_name)
                scores.append(score)
            else:
                error_models.append(model_name)

    return models, scores, error_models, zero_score_models


def _show_silhouette_message(ax, error_models, zero_score_models):
    """Display appropriate message when no valid silhouette scores are available."""
    if error_models or zero_score_models:
        # Create a more informative message about why scores couldn't be calculated
        message = "Silhouette scores could not be properly calculated\n"

        if zero_score_models:
            message += "Models with score=0: " + ", ".join(zero_score_models) + "\n"

        if error_models:
            message += "Models with errors: " + ", ".join(error_models) + "\n"

        message += "Reason: Clusters with <2 samples or calculation issues"

        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=11,
            wrap=True,
            bbox={"facecolor": "lightyellow", "alpha": 0.5, "pad": 5},
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No silhouette scores available",
            ha="center",
            va="center",
            fontsize=12,
        )


def _add_silhouette_note(ax, error_models, zero_score_models):
    """Add a note about models with errors or zero scores."""
    note_lines = []
    if zero_score_models:
        note_lines.append(f"Models with score=0: {', '.join(zero_score_models)}")
    if error_models:
        note_lines.append(f"Models with errors: {', '.join(error_models)}")
    note_lines.append("Reason: Clusters with <2 samples or calculation issues")

    note = "\n".join(note_lines)
    ax.text(
        0.5,
        -0.15,
        note,
        ha="center",
        fontsize=9,
        transform=ax.transAxes,
        bbox={"facecolor": "lightyellow", "alpha": 0.5, "pad": 5},
    )


def _plot_similarity_metrics(reports, ax):
    """
    Plot similarity metrics for each model.

    Args:
        reports: Dictionary mapping model names to their evaluation reports
        ax: Matplotlib axes object to plot on
    """
    has_similarity_data = False

    # Generate colors for models
    model_colors = get_model_colors(list(reports.keys()))

    # Track models for legend
    model_data = {}

    # Prepare data for grouped bar chart
    for i, (model_name, report) in enumerate(reports.items()):
        if "similarity_metrics" in report:
            has_similarity_data = True
            color = model_colors.get(model_name)

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

    for model_name, report in reports.items():
        if "basic_metrics" in report and "num_clusters" in report["basic_metrics"]:
            models.append(model_name)
            num_clusters.append(report["basic_metrics"]["num_clusters"])

    if not models:
        ax.text(0.5, 0.5, "No cluster count data available", ha="center", va="center")
        return

    # Generate colors for models
    model_colors = get_model_colors(models)
    colors = [model_colors[model] for model in models]

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

        # Generate colors for models
        model_colors = get_model_colors(list(reports.keys()))

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
            color = model_colors.get(model_name)

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
    # Generate colors for models
    model_colors = get_model_colors(list(reports.keys()))

    has_outlier_data = False

    for model_name, report in reports.items():
        if "outliers" not in report or not report["outliers"]:
            continue

        # Get outlier scores
        outlier_scores = list(report["outliers"].values())
        if not outlier_scores:
            continue

        has_outlier_data = True

        # Plot histogram of outlier scores with full opacity
        color = model_colors.get(model_name)
        ax.hist(outlier_scores, bins=20, alpha=1.0, color=color, label=model_name)

    if not has_outlier_data:
        ax.text(0.5, 0.5, "No outlier data available", ha="center", va="center")
        return

    ax.set_title("Outlier Score Distribution")
    ax.set_xlabel("Outlier Score")
    ax.set_ylabel("Frequency")
    ax.legend()


def visualize_evaluation_dashboard(
    reports: Dict[str, Dict[str, Any]],
    output_dir: str,
    filename: str = "evaluation_dashboard.png",
    show_plot: bool = False,
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
        show_plot: Whether to display the plot interactively

    Returns:
        Path to the saved visualization file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
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

    # Save the figure
    plt.savefig(output_path)

    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

    logger.info(f"Evaluation dashboard saved to {output_path}")
    return output_path
