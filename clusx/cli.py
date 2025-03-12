"""Command-line interface for the Clusterium.

This module provides a command-line interface for clustering text data,
benchmarking clustering results, and generating reports. It handles command-line
arguments, environment configuration, and execution of the appropriate toolkit
functionality based on user commands.

"""

import os
from pathlib import Path
from typing import Optional

import click

from clusx.logging import get_logger, setup_logging
from clusx.version import __copyright__, __version__

logger = get_logger(__name__)

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / ".cache"


def common_options(func):
    """Common options for all commands."""
    func = click.option(
        "--output-dir",
        type=click.Path(exists=True, file_okay=False, dir_okay=True),
        default=str(OUTPUT_DIR),
        help="Directory to save output files (default: ./output)",
    )(func)
    return func


@click.group(help="QA Dataset Clustering Toolkit")
@click.version_option(
    version=__version__,
    prog_name="clusx",
    message=f"""%(prog)s %(version)s
{__copyright__}
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.""",
)
def cli():
    """QA Dataset Toolkit for clustering and benchmarking."""
    pass


@cli.command(help="Cluster text data using Dirichlet Process and Pitman-Yor Process")
@common_options
@click.option(
    "--input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the input CSV file containing QA pairs",
    required=True,
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
    "--alpha",
    default=5.0,
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
    "--variance",
    default=0.1,
    show_default=True,
    type=float,
    help="Variance parameter for likelihood model",
)
@click.option(
    "--random-seed",
    default=None,
    show_default=True,
    type=int,
    help="Random seed for reproducible clustering",
)
@click.option(
    "--cache-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=CACHE_DIR,
    help="Directory to cache embeddings  [default: ./.cache]",
)
def cluster(
    input: str,
    column: str,
    output: str,
    output_dir: str,
    alpha: float,
    sigma: float,
    variance: float,
    random_seed: Optional[int],
    cache_dir: str,
) -> None:
    """Cluster text data using Dirichlet Process and Pitman-Yor Process."""
    from clusx.clustering import (
        DirichletProcess,
        EmbeddingCache,
        PitmanYorProcess,
    )
    from clusx.clustering.utils import (
        load_data_from_csv,
        save_clusters_to_csv,
        save_clusters_to_json,
    )

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
            raise click.ClickException(f"No data found in column '{column}'")

        logger.info(f"Loaded {len(texts)} texts for clustering")

        # Create cache provider
        cache_provider = EmbeddingCache(cache_dir=cache_dir)

        # Create base measure with variance
        base_measure = {"variance": variance}

        # Perform Dirichlet Process clustering
        logger.info("Performing Dirichlet Process clustering...")
        dp = DirichletProcess(
            alpha=alpha,
            base_measure=base_measure,
            cache=cache_provider,
            random_state=random_seed,
        )
        clusters_dp, cluster_params_dp = dp.fit(texts)
        logger.info(f"DP clustering complete. Found {len(set(clusters_dp))} clusters")

        # Perform Pitman-Yor Process clustering
        logger.info("Performing Pitman-Yor Process clustering...")
        pyp = PitmanYorProcess(
            alpha=alpha,
            sigma=sigma,
            base_measure=base_measure,
            cache=cache_provider,
            random_state=random_seed,
        )
        clusters_pyp, cluster_params_pyp = pyp.fit(texts)
        logger.info(f"PYP clustering complete. Found {len(set(clusters_pyp))} clusters")

        # Save results
        output_basename = os.path.basename(output)

        # Save CSV files
        dp_output = os.path.join(output_dir, output_basename.replace(".csv", "_dp.csv"))
        pyp_output = os.path.join(
            output_dir, output_basename.replace(".csv", "_pyp.csv")
        )
        save_clusters_to_csv(
            dp_output,
            texts,
            clusters_dp,
            "DP",
            alpha=alpha,
            sigma=0.0,
            variance=variance,
        )
        save_clusters_to_csv(
            pyp_output,
            texts,
            clusters_pyp,
            "PYP",
            alpha=alpha,
            sigma=sigma,
            variance=variance,
        )

        # Save JSON files
        dp_json = os.path.join(output_dir, output_basename.replace(".csv", "_dp.json"))
        pyp_json = os.path.join(
            output_dir, output_basename.replace(".csv", "_pyp.json")
        )
        save_clusters_to_json(
            dp_json,
            texts,
            clusters_dp,
            "DP",
            data,
            alpha=alpha,
            sigma=0.0,
            variance=variance,
        )
        save_clusters_to_json(
            pyp_json,
            texts,
            clusters_pyp,
            "PYP",
            data,
            alpha=alpha,
            sigma=sigma,
            variance=variance,
        )

        # Save combined results
        qa_clusters_path = os.path.join(output_dir, "qa_clusters.json")
        save_clusters_to_json(
            qa_clusters_path,
            texts,
            clusters_dp,
            "Combined",
            data,
            alpha=alpha,
            sigma=0.0,
            variance=variance,
        )
        logger.info(f"Combined clusters saved to {qa_clusters_path}")
    except Exception as e:
        logger.exception(f"Error: {e}")
        raise click.ClickException(str(e))


@cli.command(help="Evaluate clustering results")
@common_options
@click.option(
    "--input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the input CSV file containing QA pairs",
    required=True,
)
@click.option(
    "--column",
    default="question",
    show_default=True,
    help="Column name to use for clustering",
)
@click.option(
    "--dp-clusters",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to Dirichlet Process clustering results CSV",
    required=True,
)
@click.option(
    "--pyp-clusters",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to Pitman-Yor Process clustering results CSV",
    required=True,
)
@click.option(
    "--plot/--no-plot",
    default=True,
    show_default=True,
    help="Generate evaluation plots",
)
@click.option(
    "--random-seed",
    default=None,
    show_default=True,
    type=int,
    help="Random seed for reproducible evaluation",
)
@click.option(
    "--cache-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=CACHE_DIR,
    help="Directory to cache embeddings  [default: ./.cache]",
)
def evaluate(
    input: str,
    column: str,
    dp_clusters: str,
    pyp_clusters: str,
    output_dir: str,
    plot: bool,
    random_seed: Optional[int],
    cache_dir: str,
) -> None:
    """Evaluate clustering results using established metrics."""
    from clusx.clustering import EmbeddingCache
    from clusx.clustering.utils import (
        get_embeddings,
        load_cluster_assignments,
        load_data_from_csv,
    )
    from clusx.evaluation import (
        ClusterEvaluator,
        save_evaluation_report,
    )

    try:
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Loading data from {input}, using column '{column}'...")
        texts, _ = load_data_from_csv(input, column)

        if not texts:
            raise click.ClickException(f"No data found in column '{column}'")

        logger.info(f"Loaded {len(texts)} texts for evaluation")

        # Load cluster assignments
        logger.info(f"Loading DP cluster assignments from {dp_clusters}...")
        dp_cluster_assignments, dp_params = load_cluster_assignments(dp_clusters)

        logger.info(f"Loading PYP cluster assignments from {pyp_clusters}...")
        pyp_cluster_assignments, pyp_params = load_cluster_assignments(pyp_clusters)

        # Create cache provider with random seed for reproducibility
        cache_provider = EmbeddingCache(cache_dir=cache_dir)
        embeddings = get_embeddings(texts, cache_provider)

        # Evaluate DP clusters
        logger.info("Evaluating Dirichlet Process clustering...")
        dp_evaluator = ClusterEvaluator(
            texts,
            embeddings,
            dp_cluster_assignments,
            "Dirichlet",
            alpha=dp_params["alpha"],
            sigma=dp_params["sigma"],
            variance=dp_params.get("variance", 0.1),
            random_state=random_seed,
        )
        dp_report = dp_evaluator.generate_report()

        # Evaluate PYP clusters
        logger.info("Evaluating Pitman-Yor Process clustering...")
        pyp_evaluator = ClusterEvaluator(
            texts,
            embeddings,
            pyp_cluster_assignments,
            "Pitman-Yor",
            alpha=pyp_params["alpha"],
            sigma=pyp_params["sigma"],
            variance=pyp_params.get("variance", 0.1),
            random_state=random_seed,
        )
        pyp_report = pyp_evaluator.generate_report()

        reports = {
            "Dirichlet": dp_report,
            "Pitman-Yor": pyp_report,
        }

        save_evaluation_report(reports, output_dir)

        if plot:
            from clusx.visualization import visualize_evaluation_dashboard

            # Generate the dashboard visualization
            click.echo("Generating evaluation dashboard...")
            dashboard_path = visualize_evaluation_dashboard(
                reports, output_dir, show_plot=True
            )
            click.echo(f"Visualization saved to: {dashboard_path}")
            click.echo("Close the plot window to continue.")

        click.echo("Evaluation complete.")
    except Exception as e:
        raise click.ClickException(str(e))


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
