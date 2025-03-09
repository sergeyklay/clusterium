#!/usr/bin/env python3
"""Command-line interface for the QA dataset clustering toolkit.

This module provides a command-line interface for clustering QA datasets,
benchmarking clustering results, and generating reports. It handles command-line
arguments, environment configuration, and execution of the appropriate toolkit
functionality based on user commands.
"""

import logging
import os
import signal
from pathlib import Path

import click
from dotenv import load_dotenv

from qadst import ClusterBenchmarker, HDBSCANQAClusterer, __copyright__, __version__

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Load environment variables
env_file = BASE_DIR / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    logger.warning(f"No .env file found at {env_file}")
    load_dotenv()


def common_options(func):
    """Common options for all commands."""
    func = click.option(
        "--output-dir",
        type=click.Path(exists=True, file_okay=False, dir_okay=True),
        default=str(OUTPUT_DIR),
        help="Directory to save output files (default: ./output)",
    )(func)
    func = click.option(
        "--llm-model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-4o"),
        help="LLM model to use (default: gpt-4o or OPENAI_MODEL env var)",
    )(func)
    func = click.option(
        "--embedding-model",
        type=str,
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        help="Embedding model to use (default: text-embedding-3-large or "
        "OPENAI_EMBEDDING_MODEL env var)",
    )(func)
    return func


@click.group()
@click.version_option(
    version=__version__,
    prog_name="qadst",
    message=f"""%(prog)s %(version)s
{__copyright__}
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.""",
)
def cli():
    """QA Dataset Toolkit for clustering and benchmarking."""
    pass


@cli.command("cluster")
@common_options
@click.option(
    "--input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to the input CSV file containing QA pairs",
)
@click.option(
    "--filter/--no-filter",
    default=True,
    help="Filter out engineering-focused questions",
)
@click.option(
    "--min-cluster-size",
    type=int,
    default=None,
    help="Minimum size of clusters (if not provided, calculated automatically)",
)
@click.option(
    "--min-samples",
    type=int,
    default=None,
    help="HDBSCAN min_samples parameter (default: 5)",
)
@click.option(
    "--cluster-selection-epsilon",
    type=float,
    default=None,
    help="HDBSCAN cluster_selection_epsilon parameter (default: 0.3)",
)
@click.option(
    "--keep-noise/--cluster-noise",
    default=False,
    help="Keep noise points unclustered (default: False, noise points are clustered)",
)
@click.option(
    "--cluster-selection-method",
    type=click.Choice(["eom", "leaf"]),
    default="eom",
    help="HDBSCAN cluster selection method (default: eom)",
)
def cluster_command(
    output_dir,
    llm_model,
    embedding_model,
    input,
    filter,
    min_cluster_size,
    min_samples,
    cluster_selection_epsilon,
    keep_noise,
    cluster_selection_method,
):
    """Cluster QA pairs using HDBSCAN algorithm."""
    logger.info("Starting QA dataset clustering process")
    if not input:
        raise click.UsageError("Input CSV file is required")

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda sig, frame: exit(0))

    # Check if OpenAI API key is set
    if filter and not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set, disabling filtering")
        filter = False

    clusterer = HDBSCANQAClusterer(
        embedding_model_name=embedding_model,
        llm_model_name=llm_model if filter else None,
        output_dir=output_dir,
        filter_enabled=filter,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        keep_noise=keep_noise,
        cluster_selection_method=cluster_selection_method,
    )

    logger.info(f"Processing dataset from {input}")
    logger.info(f"Using embedding model: {embedding_model}")
    if filter:
        logger.info(f"Using LLM model for filtering: {llm_model}")

    result = clusterer.process_dataset(str(input))

    logger.info("Clustering complete")
    logger.info(f"Original QA pairs: {result['original_count']}")
    logger.info(f"Deduplicated QA pairs: {result['deduplicated_count']}")
    if "filtered_count" in result:
        logger.info(f"After filtering: {result['filtered_count']}")
    logger.info(f"Number of clusters: {result['num_clusters']}")
    logger.debug(f"Clusters JSON saved to: {result['json_output_path']}")
    logger.debug(f"Cleaned CSV saved to: {result['csv_output_path']}")


@cli.command("benchmark")
@common_options
@click.option(
    "--clusters",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to the JSON file containing clustering results",
)
@click.option(
    "--qa-pairs",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to the CSV file containing QA pairs",
)
@click.option(
    "--use-llm/--no-llm",
    default=True,
    help="Use LLM for generating topic labels",
)
def benchmark_command(
    output_dir, llm_model, embedding_model, clusters, qa_pairs, use_llm
):
    """Benchmark clustering results and generate reports."""
    if not clusters or not qa_pairs:
        raise click.UsageError("Both clusters and qa-pairs are required")

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda sig, frame: exit(0))

    # Check if OpenAI API key is set
    if use_llm and not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set, disabling LLM topic labeling")
        use_llm = False

    benchmarker = ClusterBenchmarker(
        embedding_model_name=embedding_model,
        llm_model_name=llm_model if use_llm else None,
        output_dir=output_dir,
    )

    # Generate report
    logger.info(f"Analyzing clusters from: {clusters}")
    logger.info(f"Using QA pairs from: {qa_pairs}")
    logger.info(f"Using embedding model: {embedding_model}")
    logger.info(f"LLM topic labeling: {'enabled' if use_llm else 'disabled'}")
    if use_llm:
        logger.info(f"Using LLM model: {llm_model}")

    report_df = benchmarker.generate_cluster_report(
        clusters_json_path=clusters,
        qa_csv_path=qa_pairs,
        use_llm_for_topics=use_llm,
    )

    # Print summary
    summary_row = report_df.iloc[-1]
    logger.info(f"Total QA pairs: {summary_row['Num_QA_Pairs']}")
    logger.info(f"Metrics: {summary_row['Topic_Label']}")
    logger.info(f"Enhanced clusters JSON with metrics: {clusters}")
    logger.info(
        f"CSV report saved to: {os.path.join(output_dir, 'cluster_quality_report.csv')}"
    )

    # Print top 5 clusters by size
    top_clusters = report_df[report_df["Cluster_ID"] != "SUMMARY"].nlargest(
        5, "Num_QA_Pairs"
    )
    logger.info("Top 5 clusters by size:")
    for _, row in top_clusters.iterrows():
        logger.info(
            f"Cluster {row['Cluster_ID']}: {row['Num_QA_Pairs']} QA pairs, "
            f"Coherence: {row['Coherence_Score']:.2f}, Topic: {row['Topic_Label']}"
        )


def main():
    """Main entry point for the QA toolkit CLI.

    Handles exceptions and returns appropriate exit codes:
    - 0: Success
    - 1: General error
    - 128+SIGINT: Keyboard interrupt (Ctrl+C)
    - 128+SIGTERM: Termination signal
    """
    try:
        cli()
        return 0
    except KeyboardInterrupt:
        logger.error("Operation interrupted by user")
        return 128 + signal.SIGINT
    except SystemExit as e:
        # Click raises SystemExit with the exit code
        return e.code if isinstance(e.code, int) else 128 + signal.SIGTERM
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
