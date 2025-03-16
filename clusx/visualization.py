"""
Visualization module for Clusterium.

This module provides functions for visualizing clustering results and evaluation
metrics.

"""

from __future__ import annotations

import os
from collections import Counter
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from clusx.errors import VisualizationError
from clusx.logging import get_logger

if TYPE_CHECKING:
    from typing import Any

    from matplotlib.axes import Axes


logger = get_logger(__name__)

MIN_DATASET_SIZE = 10
"""
Minimum dataset size for which visualizations are considered safe.

Note
----
Visualizations may not be meaningful or could be misleading when applied to datasets
smaller than this threshold.
"""

plt.style.use("default")


def get_model_colors(model_names: list[str]) -> dict[str, Any]:
    """Generate consistent colors for models using academically popular colormaps.

    Selects appropriate colormaps based on visualization best practices for clustering:

    - For typical case (≤10 models): Uses 'Set1' which provides distinct, balanced hues
      that ensure clear differentiation among groups.
    - For more models: Uses 'tab20' which provides up to 20 distinct colors, with alpha
      variation for cases beyond 20 models to maintain visual distinction.

    This approach follows standard practices in clustering visualization where
    colormap selection is based on the number of clusters to ensure optimal
    visual clarity and accessibility.

    Parameters
    ----------
    model_names : list[str]
        List of model names to generate colors for

    Returns
    -------
    dict
        Dictionary mapping model names to their assigned colors


    """
    num_models = len(model_names)

    # For typical case, use Set1 colormap
    if num_models <= 10:
        cmap = colormaps["Set1"]
        colors = [cmap(i / 9) for i in range(num_models)]
        return dict(zip(model_names, colors))

    cmap = colormaps["tab20"]
    colors = []
    for i in range(num_models):
        color_idx = i % 20  # Cycle through the 20 colors
        alpha = 1.0 if i < 20 else 0.7  # Use lower alpha for recycled colors
        color = list(cmap(color_idx / 19))
        color[3] = alpha  # Set alpha value
        colors.append(tuple(color))

    return dict(zip(model_names, colors))


def safe_plot(title: str | None = None, min_dataset_size: int = MIN_DATASET_SIZE):
    """
    Decorator for safely executing plotting functions with error handling.

    Parameters
    ----------
    title : str or None
        Title for the plot. If None, the function name will be used.
    min_dataset_size : int
        Minimum dataset size threshold for small dataset detection.
        Default is :const:`MIN_DATASET_SIZE`.

    Returns
    -------
    collections.abc.Callable
        Decorated function that handles errors and provides visual feedback.

    Examples
    --------
    >>> @safe_plot(title="My Custom Plot")
    >>> def plot_my_visualization(reports, ax):
    >>>     # Your plotting code here
    >>>     # No need for try/except blocks
    >>>     ax.plot(data)
    >>>     ax.set_title("My Plot")
    >>>
    >>> # Usage remains the same as the original function
    >>> plot_my_visualization(reports, ax)

    Notes
    -----
    - The decorated function must accept 'reports' and 'ax' as its first two
      arguments
    - The decorator automatically sets the plot title
    - For small datasets, a specific message is displayed
    - All exceptions are logged with detailed error messages
    """

    def decorator(plot_func):
        from functools import wraps

        @wraps(plot_func)
        def wrapper(reports, ax: Axes, *args, **kwargs):
            func_name = plot_func.__name__.replace("plot_", "")
            func_name = func_name.replace("_", " ").strip().title()
            plot_title = title if title is not None else func_name
            ax.set_title(plot_title)

            try:
                return plot_func(reports, ax, *args, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error plotting %s: %s", plot_title, e)

                small_dataset = is_small_dataset(reports, min_dataset_size)
                render_error_message(ax, plot_title, e, small_dataset, min_dataset_size)

                return None

        return wrapper

    return decorator


def is_small_dataset(reports: dict[str, dict[str, Any]], min_size: int) -> bool:
    """
    Check if the dataset is considered small based on the number of texts.

    Parameters
    ----------
    reports : dict
        Dictionary mapping model names to their evaluation reports.
    min_size : int
        Minimum number of texts threshold.

    Returns
    -------
    bool
        True if the dataset is considered small, False otherwise.

    Notes
    -----
    A dataset is considered small if:

    1. It's empty (no reports) or not a dictionary
    2. No reports have 'cluster_stats'
    3. No reports have 'num_texts' in their 'cluster_stats'
    4. Any report has fewer than min_size texts (assuming we have the same dataset
       for all reports)
    """
    if not reports or not isinstance(reports, dict):
        return True

    has_text_count_info = False
    for report in reports.values():
        if "cluster_stats" in report and "num_texts" in report["cluster_stats"]:
            has_text_count_info = True
            if report["cluster_stats"]["num_texts"] < min_size:
                return True

    return not has_text_count_info


def render_error_message(
    ax: Axes, plot_title: str, error, small_dataset: bool, min_size: int
):
    """
    Display appropriate error message on the plot.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to display the error message on.
    plot_title : str
        Title of the plot.
    error : Exception
        The exception that was raised.
    small_dataset : bool
        Whether the dataset is considered small.
    min_size : int
        Minimum dataset size threshold.

    Returns
    -------
    None
        This function modifies the provided axes in-place.
    """
    if small_dataset:
        message = f"Cannot generate {plot_title} for small datasets"
        details = f"(Requires at least {min_size} data points)"

    else:
        message = f"Error plotting {plot_title}"
        error_msg = str(error)
        details = error_msg[:50] + ("..." if len(error_msg) > 50 else "")

    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.text(
        0.5,
        0.4,
        details,
        ha="center",
        va="center",
        fontsize=9,
        color="gray",
    )

    ax.set_title(f"{plot_title} (Error)")

    # Reset scales if they might have been changed
    if hasattr(ax, "set_xscale") and hasattr(ax, "set_yscale"):
        try:
            ax.set_xscale("linear")
            ax.set_yscale("linear")
        except Exception:  # pylint: disable=broad-except
            pass


@safe_plot(title="Cluster Size Distribution (Log-Log Scale)")
def plot_cluster_size_distribution(reports, ax: Axes):
    """
    Plot cluster size distributions for each model.

    Parameters
    ----------
    reports : dict
        Dictionary mapping model names to their evaluation reports.
    ax :Axes
        Matplotlib axes to plot on.

    Returns
    -------
    None
        The function modifies the provided axes in-place.
    """
    # Generate colors for models
    model_colors = get_model_colors(list(reports.keys()))

    for model_name, report in reports.items():
        # Check if we have the required data
        has_cluster_stats = "cluster_stats" in report
        has_sizes = has_cluster_stats and "cluster_sizes" in report["cluster_stats"]

        if not has_sizes:
            logger.warning("Skipping %s: No cluster size distribution data", model_name)
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
        for _, size in cluster_size_dist.items():
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
            logger.warning("No valid cluster sizes for %s", model_name)

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


@safe_plot(title="Number of Clusters")
def plot_cluster_counts(reports, ax: Axes):
    """
    Plot the number of clusters for each model.

    Parameters
    ----------
    reports : dict
        Dictionary mapping model names to their evaluation reports.
    ax : Axes
        Matplotlib axes to plot on.

    Returns
    -------
    None
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
        raise VisualizationError("No cluster count data available")

    ax.bar(models, counts, color=colors)
    ax.set_xlabel("Model")
    ax.set_ylabel("Count")

    # Add value labels on top of bars
    for i, count in enumerate(counts):
        ax.text(i, count + 0.5, str(count), ha="center")


@safe_plot(title="Similarity Comparison")
def plot_similarity_metrics(reports, ax: Axes):
    """
    Plot similarity metrics for each model.

    Parameters
    ----------
    reports : dict
        Dictionary mapping model names to their evaluation reports.
    ax : Axes
        Matplotlib axes to plot on.

    Returns
    -------
    None
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
            logger.warning("No similarity metrics for %s", model_name)
            continue

        similarity_metrics = report["metrics"]["similarity"]

        if not similarity_metrics:
            logger.warning("Empty similarity metrics for %s", model_name)
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
        raise VisualizationError("Similarity metrics not available")

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

    # Calculate dynamic y-axis limit based on data
    intra_max = max(intra_values, default=0)
    inter_max = max(inter_values, default=0)
    max_value = max(intra_max, inter_max)

    # Add a small padding (25%) above the highest bar for aesthetics
    y_max = min(1.0, max_value * 1.25) if max_value > 0 else 0.1

    # Ensure minimum height for very small values
    y_max = max(y_max, 0.1)

    ax.set_ylabel("Cosine Similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(["Intra-cluster", "Inter-cluster"])
    ax.set_ylim(0, y_max)
    ax.legend()


def _get_valid_powerlaw_data(
    report, model_name
):  # pylint: disable=too-many-return-statements
    """
    Extract and validate power law data from a report.

    Parameters
    ----------
    report : dict
        Evaluation report for a model.
    model_name : str
        Name of the model.

    Returns
    -------
    tuple or None
        If valid, returns (alpha, sigma, xmin, valid_sizes, valid_frequencies).
        Returns None if the data is intentionally invalid (e.g., small dataset case).

    Raises
    ------
    VisualizationError
        If the powerlaw metrics are not available or contain invalid values.
    """
    has_metrics = "metrics" in report and "powerlaw" in report["metrics"]
    if not has_metrics:
        raise VisualizationError(f"No powerlaw metrics for {model_name}")

    powerlaw_metrics = report["metrics"]["powerlaw"]
    if not powerlaw_metrics:
        raise VisualizationError(f"Empty powerlaw metrics for {model_name}")

    # Get parameters
    alpha = powerlaw_metrics.get("alpha", None)
    sigma = powerlaw_metrics.get("sigma_error", None)
    xmin = powerlaw_metrics.get("xmin", None)

    # Check if parameters are intentionally None (small dataset case)
    if alpha is None and xmin is None:
        # This is an expected case for small datasets, not an error
        return None

    # Check for NaN values
    if (
        np.isnan(alpha)
        if alpha is not None
        else False or np.isnan(xmin) if xmin is not None else False
    ):
        raise VisualizationError(
            f"Invalid powerlaw parameters for {model_name}: alpha={alpha}, xmin={xmin}"
        )

    # Get cluster size distribution
    if "cluster_stats" not in report or "cluster_sizes" not in report["cluster_stats"]:
        raise VisualizationError(f"No cluster size distribution for {model_name}")

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
        raise VisualizationError(f"No valid sizes for powerlaw fit for {model_name}")

    return alpha, sigma, xmin, valid_sizes, valid_frequencies


def _generate_powerlaw_fit_line(
    valid_sizes, valid_frequencies, alpha, xmin, xmin_index, color, model_name
):
    """
    Generate and plot a power-law fit line.

    Parameters
    ----------
    valid_sizes : list
        List of valid cluster sizes.
    valid_frequencies : list
        List of frequencies for each size.
    alpha : float
        Power-law exponent.
    xmin : float
        Minimum value for which power-law holds.
    xmin_index : int
        Index of xmin in valid_sizes.
    color : str or tuple
        Color to use for the plot.
    model_name : str
        Name of the model for the label.

    Returns
    -------
    tuple
        (success, line_data) where success is a boolean and line_data is a tuple
        containing (x, y, color, label) or None if unsuccessful.
    """
    try:
        x = np.logspace(np.log10(xmin), np.log10(max(valid_sizes)), 50)
        y = [
            item ** (-alpha) * valid_frequencies[xmin_index] * (xmin**alpha)
            for item in x
        ]

        return True, (x, y, color, f"{model_name} (α={alpha:.2f})")
    except Exception as e:  # pylint: disable=broad-except
        raise VisualizationError(
            f"Error generating power-law fit for {model_name}: {e}"
        ) from e


def _display_no_powerlaw_message(ax: Axes, small_dataset: bool):
    """
    Display a message when power-law analysis is not available.

    Args:
        ax: Matplotlib axes to plot on
        small_dataset: Whether this is a small dataset
    """
    if small_dataset:
        message = "Power-law analysis requires more data points"
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
        ax.text(
            0.5,
            0.4,
            "Each cluster size needs multiple occurrences",
            ha="center",
            va="center",
            fontsize=9,
        )
    else:
        ax.text(0.5, 0.5, "No power-law fit data available", ha="center", va="center")
    ax.set_xscale("linear")
    ax.set_yscale("linear")


@safe_plot(title="Power-law Fit")
def plot_powerlaw_fit(reports, ax: Axes):
    """
    Plot power-law fit for cluster size distributions.

    Parameters
    ----------
    reports : dict
        Dictionary mapping model names to their evaluation reports.
    ax : Axes
        Matplotlib axes to plot on.

    Returns
    -------
    None
        The function plots directly on the provided axes.
    """
    has_powerlaw_data = False
    small_dataset = False

    # Check if we're dealing with a small dataset
    for report in reports.values():
        if "cluster_stats" in report and "num_texts" in report["cluster_stats"]:
            if report["cluster_stats"]["num_texts"] < MIN_DATASET_SIZE:
                small_dataset = True
                break

    # Generate colors for models
    model_colors = get_model_colors(list(reports.keys()))

    for model_name, report in reports.items():
        # Get and validate power law data
        result = _get_valid_powerlaw_data(report, model_name)
        if result is None:
            continue

        alpha, _, xmin, valid_sizes, valid_frequencies = result
        color = model_colors.get(model_name)

        # Plot empirical distribution
        ax.loglog(
            valid_sizes,
            valid_frequencies,
            "o",
            color=color,
            alpha=0.5,
            label=f"{model_name} (data)",
        )

        # Check if xmin is in valid_sizes
        try:
            xmin_index = valid_sizes.index(xmin)
        except ValueError:
            # xmin not in valid_sizes, use the closest value
            logger.warning(
                "xmin=%s not in valid sizes for %s, using closest value",
                xmin,
                model_name,
            )
            closest_idx = min(
                # pylint: disable=cell-var-from-loop
                range(len(valid_sizes)),
                key=lambda i: abs(valid_sizes[i] - xmin),
            )
            xmin = valid_sizes[closest_idx]
            xmin_index = closest_idx

        # Generate and plot power-law fit line
        success, fit_data = _generate_powerlaw_fit_line(
            valid_sizes, valid_frequencies, alpha, xmin, xmin_index, color, model_name
        )

        if success and fit_data is not None:
            x, y, color, label = fit_data
            # Plot fit line
            ax.loglog(x, y, "-", color=color, label=label)
            has_powerlaw_data = True

    if not has_powerlaw_data:
        _display_no_powerlaw_message(ax, small_dataset)
        return

    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Probability Density")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()


@safe_plot(title="Outlier Score Distribution")
def plot_outliers(reports, ax: Axes):
    """
    Plot outlier scores distribution.

    Parameters
    ----------
    reports : dict
        Dictionary mapping model names to their evaluation reports.
    ax : Axes
        Matplotlib axes to plot on.

    Returns
    -------
    None
    """
    has_outlier_data = False

    # Generate colors for models
    model_colors = get_model_colors(list(reports.keys()))

    for model_name, report in reports.items():
        # Check if we have outlier metrics
        has_metrics = "metrics" in report and "outliers" in report["metrics"]

        if not has_metrics or not report["metrics"]["outliers"]:
            logger.warning("No outlier metrics for %s", model_name)
            continue

        outlier_scores = report["metrics"]["outliers"]

        if not outlier_scores:
            logger.warning("Empty outlier scores for %s", model_name)
            continue

        # Extract scores
        scores = list(outlier_scores.values())

        # Plot histogram
        color = model_colors.get(model_name)
        ax.hist(scores, bins=20, alpha=1.0, label=model_name, color=color)

        has_outlier_data = True

    if not has_outlier_data:
        raise VisualizationError("No outlier data available")

    ax.set_xlabel("Outlier Score")
    ax.set_ylabel("Frequency")
    ax.legend()


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


def _show_silhouette_message(ax: Axes, error_models, zero_score_models):
    """Display appropriate message when no valid silhouette scores are available.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to display the message on.
    error_models : list
        List of models with errors.
    zero_score_models : list
        List of models with zero scores.

    Returns
    -------
    None
    """
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


def _add_silhouette_note(ax: Axes, error_models, zero_score_models):
    """Add a note about models with errors or zero scores.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to add the note to.
    error_models : list
        List of models with errors.
    zero_score_models : list
        List of models with zero scores.

    Returns
    -------
    None
    """
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


@safe_plot(title="Silhouette Score Comparison")
def plot_silhouette_scores(reports, ax: Axes):
    """Plot silhouette scores for each model.

    Parameters
    ----------
    reports : dict
        Dictionary mapping model names to their evaluation reports.
    ax : Axes
        Matplotlib axes to plot on.

    Returns
    -------
    None
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
    ax.set_xlabel("Clustering Model")
    ax.set_ylabel("Silhouette Score")
    ax.set_ylim(-1, 1)  # Silhouette score range

    # Add value labels on top of bars
    for i, score in enumerate(scores):
        ax.text(i, score + 0.05, f"{score:.4f}", ha="center")

    # If some models had errors or zero scores, add a note
    if error_models or zero_score_models:
        _add_silhouette_note(ax, error_models, zero_score_models)


def visualize_evaluation_dashboard(
    reports: dict[str, dict[str, Any]],
    output_dir: str,
    filename: str = "evaluation_dashboard.png",
    show_plot: bool = False,
) -> str:
    """Generate a comprehensive dashboard visualization of evaluation metrics.

    This creates a 3x2 grid of plots showing:

    1. Cluster size distribution (log-log scale)
    2. Silhouette score comparison
    3. Similarity metrics comparison
    4. Power-law fit visualization
    5. Outlier distribution
    6. Number of clusters comparison

    Parameters
    ----------
    reports : dict[str, dict[str, Any]]
        Dictionary mapping model names to their evaluation reports.
    output_dir : str
        Directory to save the visualization.
    filename : str
        Name of the output file.
        Default is ``evaluation_dashboard.png``
    show_plot : bool
        Whether to display the plot interactively.
        Default is ``False``.

    Returns
    -------
    str
        Path to the saved visualization file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Create figure with 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    # Plot layout visualization (3x2 grid):
    #
    #  +-------------------------+-------------------------+
    #  |                         |                         |
    #  | Cluster Size            | Number of               |
    #  | Distribution            | Clusters                |
    #  | [0,0]                   | [0,1]                   |
    #  |                         |                         |
    #  +-------------------------+-------------------------+
    #  |                         |                         |
    #  | Similarity              | Power-law               |
    #  | Metrics                 | Fit                     |
    #  | [1,0]                   | [1,1]                   |
    #  |                         |                         |
    #  +-------------------------+-------------------------+
    #  |                         |                         |
    #  | Outlier                 | Silhouette              |
    #  | Distribution            | Scores                  |
    #  | [2,0]                   | [2,1]                   |
    #  |                         |                         |
    #  +-------------------------+-------------------------+
    #
    plot_cluster_size_distribution(reports, axes[0, 0])
    plot_cluster_counts(reports, axes[0, 1])
    plot_similarity_metrics(reports, axes[1, 0])
    plot_powerlaw_fit(reports, axes[1, 1])
    plot_outliers(reports, axes[2, 0])
    plot_silhouette_scores(reports, axes[2, 1])

    plt.tight_layout()

    if is_small_dataset(reports, MIN_DATASET_SIZE):
        plt.subplots_adjust(bottom=0.08)
        warning_text = (
            f"Small dataset (fewer than {MIN_DATASET_SIZE} data points).\n"
            "Some visualizations may not be available or may not accurately "
            "represent the data patterns."
        )

        fig.text(0.055, 0.035, warning_text, ha="left", va="bottom", fontsize=11)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.debug("Evaluation dashboard saved to %s", output_path)

    if show_plot:
        plt.show()

    plt.close(fig)
    return output_path
