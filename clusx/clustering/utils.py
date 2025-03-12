"""
Utility functions for data loading, saving, and visualization.
"""

import csv
import json
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from clusx.logging import get_logger

logger = get_logger(__name__)


def load_data_from_csv(
    csv_file: str, column: str = "question", answer_column: str = "answer"
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Load text data from a CSV file.

    Args:
        csv_file: Path to the CSV file
        column: Column name containing the text data (default: "question")
        answer_column: Column name containing the answers (default: "answer")

    Returns:
        tuple[list[str], list[dict[str, str]]]: A tuple containing (texts, data_rows)
    """
    texts = []
    data = []  # Store full data including answers

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if column in row and row[column].strip():
                texts.append(row[column])
                # Store the full row data
                data.append(row)

    return texts, data


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
    logger.info(f"Clustering results saved to {output_file}")


def save_clusters_to_json(
    output_file: str,
    texts: list[str],
    clusters: list[int],
    model_name: str,
    data: Optional[list[dict[str, Any]]] = None,
    answer_column: str = "answer",
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
        data: List of data rows containing answers (optional)
        answer_column: Column name containing the answers (default: "answer")
        alpha: Concentration parameter (default: 1.0)
        sigma: Discount parameter (default: 0.0)
        variance: Variance parameter for likelihood model (default: 0.1)
    """
    cluster_groups = {}
    data_map = {}

    # Create a mapping from question to data row if data is provided
    if data:
        for row in data:
            if "question" in row:
                data_map[row["question"]] = row

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

        # Get the answer for the representative text
        representative_answer = "No answer available"
        if (
            data_map
            and representative_text in data_map
            and answer_column in data_map[representative_text]
        ):
            representative_answer = data_map[representative_text][answer_column]
        else:
            representative_answer = f"Answer for cluster {i + 1} using {model_name}"

        # Create the cluster object
        cluster_obj = {
            "id": i + 1,
            "representative": [
                {
                    "question": representative_text,
                    "answer": representative_answer,
                }
            ],
            "source": [],
        }

        # Add sources with their real answers if available
        for text in cluster_texts:
            answer = f"Answer for question in cluster {i + 1}"
            if data_map and text in data_map and answer_column in data_map[text]:
                answer = data_map[text][answer_column]

            cluster_obj["source"].append({"question": text, "answer": answer})

        clusters_json["clusters"].append(cluster_obj)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clusters_json, f, indent=2, ensure_ascii=False)

    logger.info(f"JSON clusters saved to {output_file}")


def get_embeddings(texts: list[str], cache_provider) -> np.ndarray:
    """
    Get embeddings for a list of texts.

    Args:
        texts: List of text strings
        cache_provider: Cache provider for embeddings

    Returns:
        Numpy array of embeddings
    """
    from clusx.clustering import DirichletProcess

    logger.info("Computing embeddings for evaluation...")
    # Use default parameters for embedding generation only
    dp = DirichletProcess(
        alpha=1.0, base_measure={"variance": 0.1}, cache=cache_provider
    )
    embeddings = []

    for text in texts:
        embedding = dp.get_embedding(text)
        # Check if the embedding is a PyTorch tensor or NumPy array
        if hasattr(embedding, "cpu"):
            # It's a PyTorch tensor
            embedding = embedding.cpu().numpy()
        elif hasattr(embedding, "numpy"):
            # It's a tensor with numpy method
            embedding = embedding.numpy()
        # If it's already a NumPy array, we can use it directly
        embeddings.append(embedding)

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
    except Exception as e:
        logger.warning(f"Error loading parameters from JSON: {e}")

    return params
