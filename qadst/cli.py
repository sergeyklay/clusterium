"""
Command-line interface for QA Dataset Clustering.
"""

import os
import sys
from typing import Optional

import click
import matplotlib.pyplot as plt

from qadst.clustering import (
    DirichletProcess,
    EmbeddingCache,
    PitmanYorProcess,
)
from qadst.clustering.utils import (
    load_data_from_csv,
    plot_cluster_distribution,
    save_clusters_to_csv,
    save_clusters_to_json,
)
from qadst.logging import get_logger, setup_logging

logger = get_logger(__name__)


@click.group(help="QA Dataset Clustering Toolkit")
def cli():
    """QA Dataset Clustering Toolkit command-line interface."""
    pass


@cli.command(help="Cluster text data using Dirichlet Process and Pitman-Yor Process")
@click.option(
    "--input",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to input CSV file",
)
@click.option(
    "--column",
    default="question",
    show_default=True,
    help="Column name to use for clustering",
)
@click.option(
    "--output",
    default="clusters_output.csv",
    show_default=True,
    help="Output CSV file path",
)
@click.option(
    "--output-dir",
    default="output",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Directory to save output files",
)
@click.option(
    "--alpha",
    default=1.0,
    show_default=True,
    type=float,
    help="Concentration parameter",
)
@click.option(
    "--sigma",
    default=0.5,
    show_default=True,
    type=float,
    help="Discount parameter for Pitman-Yor",
)
@click.option(
    "--plot/--no-plot",
    default=False,
    show_default=True,
    help="Generate cluster distribution plot",
)
@click.option(
    "--cache-dir",
    default=".cache",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Directory to cache embeddings",
)
def cluster(
    input: str,
    column: str,
    output: str,
    output_dir: str,
    alpha: float,
    sigma: float,
    plot: bool,
    cache_dir: str,
) -> None:
    """Cluster text data using Dirichlet Process and Pitman-Yor Process."""
    try:
        # Create necessary directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        logger.info(f"Loading data from {input}, using column '{column}'...")
        texts, data = load_data_from_csv(input, column)

        if not texts:
            logger.error(
                f"No data found in column '{column}'. Please check your CSV file."
            )
            sys.exit(1)

        logger.info(f"Loaded {len(texts)} texts for clustering")

        # Create cache provider
        cache_provider = EmbeddingCache(cache_dir=cache_dir)

        # Perform Dirichlet Process clustering
        logger.info("Performing Dirichlet Process clustering...")
        dp = DirichletProcess(alpha=alpha, base_measure=None, cache=cache_provider)
        clusters_dp, params_dp = dp.fit(texts)
        logger.info(f"DP clustering complete. Found {len(set(clusters_dp))} clusters")

        # Perform Pitman-Yor Process clustering
        logger.info("Performing Pitman-Yor Process clustering...")
        pyp = PitmanYorProcess(
            alpha=alpha,
            sigma=sigma,
            base_measure=None,
            cache=cache_provider,
        )
        clusters_pyp, params_pyp = pyp.fit(texts)
        logger.info(f"PYP clustering complete. Found {len(set(clusters_pyp))} clusters")

        # Save results
        output_basename = os.path.basename(output)

        # Save CSV files
        dp_output = os.path.join(output_dir, output_basename.replace(".csv", "_dp.csv"))
        pyp_output = os.path.join(
            output_dir, output_basename.replace(".csv", "_pyp.csv")
        )
        save_clusters_to_csv(dp_output, texts, clusters_dp, "DP")
        save_clusters_to_csv(pyp_output, texts, clusters_pyp, "PYP")

        # Save JSON files
        dp_json = os.path.join(output_dir, output_basename.replace(".csv", "_dp.json"))
        pyp_json = os.path.join(
            output_dir, output_basename.replace(".csv", "_pyp.json")
        )
        save_clusters_to_json(dp_json, texts, clusters_dp, "DP", data)
        save_clusters_to_json(pyp_json, texts, clusters_pyp, "PYP", data)

        # Save combined results
        qa_clusters_path = os.path.join(output_dir, "qa_clusters.json")
        save_clusters_to_json(qa_clusters_path, texts, clusters_dp, "Combined", data)
        logger.info(f"Combined clusters saved to {qa_clusters_path}")

        # Generate plot if requested
        if plot:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plot_cluster_distribution(
                clusters_dp, "Dirichlet Process Cluster Sizes", "blue"
            )
            plt.subplot(1, 2, 2)
            plot_cluster_distribution(
                clusters_pyp, "Pitman-Yor Process Cluster Sizes", "red"
            )
            plt.tight_layout()
            plot_path = os.path.join(
                output_dir, output_basename.replace(".csv", "_clusters.png")
            )
            plt.savefig(plot_path)
            logger.info(f"Cluster distribution plot saved to {plot_path}")
            plt.show()

    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)


# Example of how to add another command in the future:
# @cli.command(help="Benchmark clustering results")
# @click.option(
#     "--clusters",
#     required=True,
#     type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
#     help="Path to clusters JSON file",
# )
# def benchmark(clusters: str) -> None:
#     """Benchmark clustering results."""
#     logger.info(f"Benchmarking clusters from {clusters}...")
#     # Implementation goes here


def main(args: Optional[list[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Set up logging
    setup_logging()

    try:
        # Invoke the Click command
        cli.main(args=args, standalone_mode=False)
        return 0
    except click.exceptions.Abort:
        # Handle keyboard interrupts gracefully
        logger.warning("Operation aborted by user")
        return 130  # Standard exit code for SIGINT
    except click.exceptions.Exit as e:
        # Handle normal exit
        return e.exit_code
    except Exception as e:
        # Handle unexpected errors
        logger.exception(f"Unexpected error: {e}")
        return 1
