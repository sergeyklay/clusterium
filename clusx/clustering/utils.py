"""
Utility functions for data loading, saving, and visualization.
"""

from __future__ import annotations

import csv
import json
import os
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from clusx.logging import get_logger
from clusx.utils import to_numpy

if TYPE_CHECKING:
    from typing import Optional, Union

    import torch
    from numpy.typing import NDArray

    EmbeddingTensor = Union[torch.Tensor, NDArray[np.float32]]


logger = get_logger(__name__)


def load_data(input_file: str, column: Optional[str] = None) -> list[str]:
    """
    Load text data from a file. Supports text files and CSV files.

    Args:
        input_file: Path to the input file (text or CSV)
        column: Column name containing the text data (required for CSV files)

    Returns:
        list[str]: A list of texts

    Raises:
        ValueError: If a CSV file is provided without specifying a column
    """
    texts = []

    # Check if the file is a CSV by trying to parse it
    is_csv = False
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            # Read the first line and check if it contains a comma
            first_line = f.readline().strip()
            if "," in first_line:
                # Try to parse as CSV
                sniffer = csv.Sniffer()
                # Just check if it can be parsed as CSV, don't need the dialect
                sniffer.sniff(first_line)
                is_csv = True
    except (OSError, csv.Error):
        # If any error occurs, assume it's a text file
        is_csv = False

    if is_csv:
        if column is None:
            raise ValueError("Column name must be specified when using a CSV file")

        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if column in row and row[column].strip():
                    texts.append(row[column])
    else:
        # Process as a text file (one text per line)
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    texts.append(line)

    return texts


def save_clusters_to_csv(
    output_file: str,
    texts: list[str],
    clusters: list[int],
    model_name: str,
    alpha: float = 1.0,
    sigma: float = 0.0,
    variance: float = 0.1,
) -> None:
    """
    Save clustering results to a CSV file.

    Args:
        output_file: Path to the output CSV file
        texts: List of text strings
        clusters: List of cluster assignments
        model_name: Name of the clustering model
        alpha: Concentration parameter (default: 1.0)
        sigma: Discount parameter (default: 0.0)
        variance: Variance parameter for likelihood model (default: 0.1)
    """
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Add parameters as metadata in the header
        writer.writerow(["Text", f"Cluster_{model_name}", "Alpha", "Sigma", "Variance"])
        for text, cluster in zip(texts, clusters):
            writer.writerow([text, cluster, alpha, sigma, variance])
    logger.info("Clustering results saved to %s", output_file)


def save_clusters_to_json(
    output_file: str,
    texts: list[str],
    clusters: list[int],
    model_name: str,
    alpha: float = 1.0,
    sigma: float = 0.0,
    variance: float = 0.1,
) -> None:
    """
    Save clustering results to a JSON file.

    Args:
        output_file: Path to the output JSON file
        texts: List of text strings
        clusters: List of cluster assignments
        model_name: Name of the clustering model
        alpha: Concentration parameter (default: 1.0)
        sigma: Discount parameter (default: 0.0)
        variance: Variance parameter for likelihood model (default: 0.1)
    """
    cluster_groups = {}

    # Group texts by cluster
    for text, cluster_id in zip(texts, clusters):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(text)

    clusters_json = {
        "clusters": [],
        "metadata": {
            "model_name": model_name,
            "alpha": alpha,
            "sigma": sigma,
            "variance": variance,
        },
    }

    for i, (cluster_id, cluster_texts) in enumerate(cluster_groups.items()):
        representative_text = cluster_texts[0]

        # Create the cluster object with the new format
        cluster_obj = {
            "id": i + 1,
            "representative": representative_text,
            "members": cluster_texts,
        }

        clusters_json["clusters"].append(cluster_obj)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clusters_json, f, indent=2, ensure_ascii=False)

    logger.info("JSON clusters saved to %s", output_file)


def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts.

    Args:
        texts: List of text strings

    Returns:
        Numpy array of embeddings
    """
    from clusx.clustering import DirichletProcess

    logger.info("Computing embeddings for evaluation...")
    # Use default parameters for embedding generation only
    dp = DirichletProcess(alpha=1.0)
    embeddings = []

    # Process texts with progress bar
    for text in tqdm(texts, desc="Computing embeddings", total=len(texts)):
        emb_array = to_numpy(dp.get_embedding(text))
        embeddings.append(emb_array)

    return np.array(embeddings)


def load_cluster_assignments(csv_path: str) -> tuple[list[int], dict[str, float]]:
    """
    Load cluster assignments and parameters from a CSV file.

    Args:
        csv_path: Path to the CSV file containing cluster assignments

    Returns:
        tuple[list[int], dict[str, float]]: A tuple containing:
            - List of cluster assignments
            - Dictionary of parameters (alpha, sigma, variance)

    Raises:
        ValueError: If no cluster column is found in the CSV file
    """
    df = pd.read_csv(csv_path)

    # Check which column contains the cluster assignments
    cluster_column = None
    for col in df.columns:
        if col.lower().startswith("cluster"):
            cluster_column = col
            break

    if not cluster_column:
        raise ValueError(f"No cluster column found in {csv_path}")

    # Extract cluster assignments
    cluster_assignments = df[cluster_column].tolist()

    # Extract parameters from file content if available
    params = {"alpha": 1.0, "sigma": 0.0, "variance": 0.1}  # Default values

    # Check if parameter columns exist in the CSV
    if "Alpha" in df.columns:
        params["alpha"] = float(df["Alpha"].iloc[0])
    if "Sigma" in df.columns:
        params["sigma"] = float(df["Sigma"].iloc[0])
    if "Variance" in df.columns:
        params["variance"] = float(df["Variance"].iloc[0])
    else:
        # Fallback to extracting from filename if columns don't exist
        # (for backward compatibility)
        filename = os.path.basename(csv_path)

        # Look for alpha in filename (e.g., alpha_1.0)
        alpha_match = re.search(r"alpha[_-](\d+\.\d+)", filename)
        if alpha_match:
            params["alpha"] = float(alpha_match.group(1))

        # Look for sigma in filename (e.g., sigma_0.5)
        sigma_match = re.search(r"sigma[_-](\d+\.\d+)", filename)
        if sigma_match:
            params["sigma"] = float(sigma_match.group(1))

        # Look for variance in filename (e.g., var_0.1)
        var_match = re.search(r"var[_-](\d+\.\d+)", filename)
        if var_match:
            params["variance"] = float(var_match.group(1))

    return cluster_assignments, params


def load_parameters_from_json(json_path: str) -> dict[str, float]:
    """
    Load clustering parameters from a JSON file.

    Args:
        json_path: Path to the JSON file containing clustering results

    Returns:
        dict[str, float]: A dictionary of parameters (alpha, sigma)
    """
    params = {"alpha": 1.0, "sigma": 0.0}  # Default values

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if metadata is available in the JSON
        if "metadata" in data:
            if "alpha" in data["metadata"]:
                params["alpha"] = float(data["metadata"]["alpha"])
            if "sigma" in data["metadata"]:
                params["sigma"] = float(data["metadata"]["sigma"])
    except (OSError, json.JSONDecodeError) as err:
        logger.error("Error loading parameters from JSON: %s", err)

    return params
