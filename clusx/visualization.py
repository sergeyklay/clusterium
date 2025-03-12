"""
Visualization module for QA Dataset Clustering.

This module provides functions for visualizing clustering results and evaluation
metrics.
"""

import os
import textwrap
from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from .logging import get_logger

logger = get_logger(__name__)

plt.style.use("default")


def get_model_colors(model_names: list[str]) -> dict[str, Any]:
    """Generate consistent colors for models using academically popular colormaps.

    Uses 'Set1' and 'tab10' colormaps which are standard in academic publications.

    Args:
        model_names: List of model names to generate colors for

    Returns:
        Dictionary mapping model names to their assigned colors
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
        has_cluster_stats = "cluster_stats" in report
        has_sizes = has_cluster_stats and "cluster_sizes" in report["cluster_stats"]

        if not has_sizes:
            logger.warning(f"Skipping {model_name}: No cluster size distribution data")
            continue

        # Use pre-computed cluster size distribution
        cluster_size_dist = report["cluster_stats"]["cluster_sizes"]

        # Get clustering parameters
        alpha = report["parameters"].get("alpha", "N/A")
        sigma = report["parameters"].get("sigma", "N/A")

        # Create label with model name and parameters
        label = f"{model_name} (α={alpha}, σ={sigma})"

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
        if valid_sizes and valid_frequencies:
            ax.loglog(
                valid_sizes,
                valid_frequencies,
                marker="o",
                linestyle="--",
                label=label,
                color=color,
                alpha=0.7,
            )
        else:
            logger.warning(f"No valid cluster sizes for {model_name}")

    ax.set_title("Cluster Size Distribution (Log-Log Scale)")
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Number of Clusters")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()

    # If no data was plotted, show a message
    if not ax.get_lines():
        ax.text(
            0.5,
            0.5,
            "No cluster count data available",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_xscale("linear")
        ax.set_yscale("linear")


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
        # Check if we have metrics and silhouette score
        if "metrics" in report and "silhouette_score" in report["metrics"]:
            score = report["metrics"]["silhouette_score"]
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

    # Prepare data for grouped bar chart
    model_names = []
    intra_values = []
    inter_values = []
    colors = []

    for model_name, report in reports.items():
        # Check if we have similarity metrics
        has_metrics = "metrics" in report and "similarity" in report["metrics"]

        if not has_metrics:
            logger.warning(f"No similarity metrics for {model_name}")
            continue

        similarity_metrics = report["metrics"]["similarity"]

        if not similarity_metrics:
            logger.warning(f"Empty similarity metrics for {model_name}")
            continue

        # Extract metrics
        intra_sim = similarity_metrics.get("intra_cluster_similarity", 0)
        inter_sim = similarity_metrics.get("inter_cluster_similarity", 0)

        # Store for plotting
        model_names.append(model_name)
        intra_values.append(intra_sim)
        inter_values.append(inter_sim)
        colors.append(model_colors.get(model_name))

        has_similarity_data = True

    if not has_similarity_data:
        ax.text(0.5, 0.5, "Similarity metrics not available", ha="center", va="center")
        return

    # Set up positions for the bars
    x = np.arange(2)  # Two groups: intra and inter
    width = 0.8 / len(model_names)  # Width of each bar, adjusted for number of models

    # Plot bars for each model
    for i, (model, intra, inter, color) in enumerate(
        zip(model_names, intra_values, inter_values, colors)
    ):
        # Calculate position offset for this model's bars
        offset = i * width - (len(model_names) - 1) * width / 2

        # Plot the bars
        ax.bar(x[0] + offset, intra, width, label=model, color=color)
        ax.bar(x[1] + offset, inter, width, color=color, alpha=1.0)

    # Set labels
    ax.set_title("Similarity Comparison")
    ax.set_ylabel("Cosine Similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(["Intra-cluster", "Inter-cluster"])
    ax.set_ylim(0, 1)
    ax.legend()


def _plot_cluster_counts(reports, ax):
    """
    Plot the number of clusters for each model.

    Args:
        reports: Dictionary mapping model names to their evaluation reports
        ax: Matplotlib axes object to plot on
    """
    # Generate colors for models
    model_colors = get_model_colors(list(reports.keys()))

    models = []
    counts = []
    colors = []

    for model_name, report in reports.items():
        if "cluster_stats" in report and "num_clusters" in report["cluster_stats"]:
            models.append(model_name)
            counts.append(report["cluster_stats"]["num_clusters"])
            colors.append(model_colors.get(model_name))

    if not models:
        ax.text(0.5, 0.5, "No cluster count data available", ha="center", va="center")
        return

    ax.bar(models, counts, color=colors)
    ax.set_title("Number of Clusters")
    ax.set_xlabel("Model")
    ax.set_ylabel("Count")

    # Add value labels on top of bars
    for i, count in enumerate(counts):
        ax.text(i, count + 0.5, str(count), ha="center")


def _plot_powerlaw_fit(reports, ax):
    """
    Plot power-law fit for cluster size distributions.

    Args:
        reports: Dictionary mapping model names to their evaluation reports
        ax: Matplotlib axes object to plot on
    """
    has_powerlaw_data = False

    # Generate colors for models
    model_colors = get_model_colors(list(reports.keys()))

    for model_name, report in reports.items():
        # Check if we have powerlaw metrics
        has_metrics = "metrics" in report and "powerlaw" in report["metrics"]

        if not has_metrics:
            logger.warning(f"No powerlaw metrics for {model_name}")
            continue

        powerlaw_metrics = report["metrics"]["powerlaw"]

        if not powerlaw_metrics:
            logger.warning(f"Empty powerlaw metrics for {model_name}")
            continue

        # Get parameters
        alpha = powerlaw_metrics.get("alpha", None)
        sigma = powerlaw_metrics.get("sigma_error", None)
        xmin = powerlaw_metrics.get("xmin", None)

        if alpha is None or xmin is None:
            logger.warning(f"Missing powerlaw parameters for {model_name}")
            continue

        # Get cluster size distribution
        if (
            "cluster_stats" not in report
            or "cluster_sizes" not in report["cluster_stats"]
        ):
            logger.warning(f"No cluster size distribution for {model_name}")
            continue

        cluster_sizes = report["cluster_stats"]["cluster_sizes"]

        # Convert to frequency distribution
        size_frequency = Counter(cluster_sizes.values())

        # Convert to lists for plotting
        sizes = sorted(size_frequency.keys())
        frequencies = [size_frequency[size] for size in sizes]

        # Filter out zeros for log scale
        valid_indices = [
            i for i, freq in enumerate(frequencies) if freq > 0 and sizes[i] > 0
        ]
        valid_sizes = [sizes[i] for i in valid_indices]
        valid_frequencies = [frequencies[i] for i in valid_indices]

        if not valid_sizes:
            logger.warning(f"No valid sizes for powerlaw fit for {model_name}")
            continue

        # Plot empirical distribution
        color = model_colors.get(model_name)
        ax.loglog(
            valid_sizes,
            valid_frequencies,
            "o",
            color=color,
            alpha=0.5,
            label=f"{model_name} (data)",
        )

        # Generate power-law fit line
        x = np.logspace(np.log10(xmin), np.log10(max(valid_sizes)), 50)
        y = [
            item ** (-alpha)
            * valid_frequencies[valid_sizes.index(xmin)]
            * (xmin**alpha)
            for item in x
        ]

        # Plot fit line
        ax.loglog(
            x, y, "-", color=color, label=f"{model_name} (α={alpha:.2f}±{sigma:.2f})"
        )

        has_powerlaw_data = True

    if not has_powerlaw_data:
        ax.text(0.5, 0.5, "No power-law fit data available", ha="center", va="center")
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        return

    ax.set_title("Power-law Fit")
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Probability Density")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()


def _plot_outliers(reports, ax):
    """
    Plot outlier scores distribution.

    Args:
        reports: Dictionary mapping model names to their evaluation reports
        ax: Matplotlib axes object to plot on
    """
    has_outlier_data = False

    # Generate colors for models
    model_colors = get_model_colors(list(reports.keys()))

    for model_name, report in reports.items():
        # Check if we have outlier metrics
        has_metrics = "metrics" in report and "outliers" in report["metrics"]

        if not has_metrics or not report["metrics"]["outliers"]:
            logger.warning(f"No outlier metrics for {model_name}")
            continue

        outlier_scores = report["metrics"]["outliers"]

        if not outlier_scores:
            logger.warning(f"Empty outlier scores for {model_name}")
            continue

        # Extract scores
        scores = list(outlier_scores.values())

        # Plot histogram
        color = model_colors.get(model_name)
        ax.hist(scores, bins=20, alpha=1.0, label=model_name, color=color)

        has_outlier_data = True

    if not has_outlier_data:
        ax.text(0.5, 0.5, "No outlier data available", ha="center", va="center")
        return

    ax.set_title("Outlier Score Distribution")
    ax.set_xlabel("Outlier Score")
    ax.set_ylabel("Frequency")
    ax.legend()


def visualize_evaluation_dashboard(
    reports: dict[str, dict[str, Any]],
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

    # 2. Number of clusters (top-right)
    _plot_cluster_counts(reports, axes[0, 1])

    # 3. Similarity metrics (middle-left)
    _plot_similarity_metrics(reports, axes[1, 0])

    # 4. Power-law fit (middle-right)
    _plot_powerlaw_fit(reports, axes[1, 1])

    # 5. Outlier distribution (bottom-left)
    _plot_outliers(reports, axes[2, 0])

    # 6. Silhouette score comparison (bottom-right)
    _plot_silhouette_scores(reports, axes[2, 1])

    plt.tight_layout()

    # Add a methodological note at the bottom of the figure
    note = (
        "Methodological note: α in clustering models (DP, PYP) represents the "
        "concentration parameter controlling cluster formation propensity, while α "
        "in power-law analysis denotes the scaling exponent of the cluster size "
        "distribution. σ in PYP is the discount parameter governing power-law "
        "behavior, whereas σ-error quantifies uncertainty in the power-law exponent "
        "estimate."
    )

    # Manually wrap the text to ensure it fits within the figure width
    wrapped_note = "\n".join(textwrap.wrap(note, width=100))

    # Add some extra space at the bottom for the note
    plt.subplots_adjust(bottom=0.08)

    # Create text box with academic styling
    fig.text(
        0.05,
        0.01,
        wrapped_note,
        ha="left",
        va="bottom",
        fontsize=9,
        fontstyle="italic",
        bbox=dict(
            boxstyle="round",
            facecolor="#f8f8f8",
            edgecolor="#cccccc",
            alpha=0.95,
            pad=0.7,
        ),
    )

    # Save the figure
    plt.savefig(output_path)

    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

    logger.info(f"Evaluation dashboard saved to {output_path}")
    return output_path
