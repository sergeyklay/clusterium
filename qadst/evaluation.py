"""
Evaluation module for QA Dataset Clustering.

This module provides tools for evaluating the quality and consistency of clusters
generated by the clustering algorithms. It implements established metrics for
cluster validation in the context of text data clustering.
"""

import json
import os
from typing import Any, Dict, List, Union

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from qadst.logging import get_logger

logger = get_logger(__name__)


# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy data types.

    This encoder converts NumPy types to their Python equivalents:
    - numpy.ndarray -> list
    - numpy.float32, numpy.float64 -> float
    - numpy.int32, numpy.int64 -> int
    - numpy.bool_ -> bool
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):  # type: ignore
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):  # type: ignore
            return int(obj)
        if isinstance(obj, np.bool_):  # type: ignore
            return bool(obj)
        if isinstance(obj, bool):
            return bool(obj)
        try:
            # Try to convert any other NumPy type to a Python type
            return obj.item() if hasattr(obj, "item") else obj
        except (AttributeError, ValueError, TypeError):
            return super().default(obj)


class ClusterEvaluator:
    """
    Evaluates the quality of text clusters using established metrics.

    This class provides methods to assess cluster quality using metrics like
    silhouette score, which measures how similar an object is to its own cluster
    compared to other clusters.

    Note on parameters:
    - alpha, sigma: Input parameters used in the clustering algorithms
      (Dirichlet Process and Pitman-Yor Process)

    - In detect_powerlaw_distribution():
      - alpha: Output parameter representing the power law exponent
      - sigma_error: Standard error of the power law alpha estimate

    These parameters share names but represent different concepts.
    """

    def __init__(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        cluster_assignments: List[int],
        model_name: str,
        alpha: float = 1.0,
        sigma: float = 0.0,
    ):
        """
        Initialize the cluster evaluator.

        Args:
            texts: List of text strings that were clustered
            embeddings: Numpy array of embeddings for each text
            cluster_assignments: List of cluster IDs for each text
            model_name: Name of the clustering model (e.g., "DP", "PYP")
            alpha: Concentration parameter (default: 1.0)
            sigma: Discount parameter for Pitman-Yor Process (default: 0.0)
        """
        self.texts = texts
        self.embeddings = embeddings
        self.cluster_assignments = cluster_assignments
        self.model_name = model_name
        self.alpha = alpha
        self.sigma = sigma
        self.unique_clusters = sorted(set(cluster_assignments))

        # Validate inputs
        if len(texts) != len(embeddings) or len(texts) != len(cluster_assignments):
            raise ValueError(
                "Length mismatch: texts, embeddings, and cluster_assignments "
                "must have the same length"
            )

        logger.info(
            f"Initialized cluster evaluator for {model_name} with {len(texts)} texts "
            f"and {len(self.unique_clusters)} clusters"
        )

    def calculate_silhouette_score(self) -> Union[float, int]:
        """
        Calculate the silhouette score for the clustering.

        The silhouette score measures how similar an object is to its own cluster
        compared to other clusters. The score ranges from -1 to 1, where:
        - A high value (close to 1) indicates the object is well matched to its cluster
        - A value near 0 indicates the object is on or very close to the decision
          boundary
        - A negative value indicates the object might be assigned to the wrong cluster

        Returns:
            Silhouette score as a float
        """
        # We need at least 2 clusters and each cluster must have at least 2 samples
        if len(self.unique_clusters) < 2:
            logger.warning(
                f"Cannot calculate silhouette score: only "
                f"{len(self.unique_clusters)} cluster found"
            )
            return 0.0

        # Count samples per cluster
        cluster_counts = {}
        for cluster_id in self.cluster_assignments:
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        # Check if any cluster has only one sample
        single_sample_clusters = [c for c, count in cluster_counts.items() if count < 2]
        if single_sample_clusters:
            logger.warning(
                f"Cannot calculate silhouette score: "
                f"{len(single_sample_clusters)} clusters have fewer than 2 samples"
            )
            return 0.0

        try:
            score = silhouette_score(
                self.embeddings, self.cluster_assignments, metric="cosine"
            )
            logger.info(f"Silhouette score for {self.model_name}: {score:.4f}")
            return float(score)
        except Exception as e:
            logger.error(f"Error calculating silhouette score: {e}")
            return 0.0

    def calculate_similarity_metrics(self) -> Dict[str, Union[float, np.floating]]:
        """
        Calculate intra-cluster and inter-cluster similarity metrics.

        Returns:
            Dictionary containing similarity metrics
        """
        try:
            # Initialize containers for similarities
            intra_cluster_sims = []
            inter_cluster_sims = []

            # Calculate similarities for each cluster
            for cluster_id in self.unique_clusters:
                # Get indices for this cluster
                cluster_indices = [
                    i for i, c in enumerate(self.cluster_assignments) if c == cluster_id
                ]
                other_indices = [
                    i for i, c in enumerate(self.cluster_assignments) if c != cluster_id
                ]

                if len(cluster_indices) <= 1:
                    # Skip clusters with only one element (no intra-similarity)
                    continue

                # Get embeddings for this cluster and others
                cluster_embeds = self.embeddings[cluster_indices]

                # Calculate intra-cluster similarity (within cluster)
                # Average of pairwise similarities within the cluster
                similarities = cosine_similarity(cluster_embeds)
                # Exclude self-similarities (diagonal)
                np.fill_diagonal(similarities, 0)
                # Average similarity
                avg_sim = similarities.sum() / (similarities.size - len(similarities))
                intra_cluster_sims.append(avg_sim)

                # Calculate inter-cluster similarity (between clusters)
                if other_indices:
                    other_embeds = self.embeddings[other_indices]
                    # Average similarity between this cluster and all others
                    inter_sim = np.mean(cosine_similarity(cluster_embeds, other_embeds))
                    inter_cluster_sims.append(inter_sim)

            # Calculate overall metrics
            metrics = {
                "intra_cluster_mean": (
                    np.mean(intra_cluster_sims) if intra_cluster_sims else 0.0
                ),
                "inter_cluster_mean": (
                    np.mean(inter_cluster_sims) if inter_cluster_sims else 0.0
                ),
                "separation": (
                    (np.mean(intra_cluster_sims) - np.mean(inter_cluster_sims))
                    if (intra_cluster_sims and inter_cluster_sims)
                    else 0.0
                ),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating similarity metrics: {e}")
            return {
                "intra_cluster_mean": 0.0,
                "inter_cluster_mean": 0.0,
                "separation": 0.0,
            }

    def detect_powerlaw_distribution(self) -> Dict[str, Any]:
        """
        Detect if cluster sizes follow a power-law distribution.

        Note: The alpha parameter returned by this method is the power law exponent,
        which is different from the alpha concentration parameter used in the
        Dirichlet Process and Pitman-Yor Process clustering algorithms.

        Similarly, the sigma_error parameter is the standard error of the power law
        alpha estimate, not the sigma discount parameter used in Pitman-Yor Process.

        Returns:
            Dictionary containing:
                - alpha: Power law exponent (not the clustering alpha parameter)
                - sigma_error: Standard error of the alpha estimate
                - xmin: Minimum x value for which the power law applies
                - is_powerlaw: Boolean indicating if distribution follows a power law
        """
        try:
            from collections import Counter

            import powerlaw  # type: ignore

            sizes = list(Counter(self.cluster_assignments).values())
            if len(sizes) < 5:
                return {"alpha": None, "sigma_error": None, "is_powerlaw": False}

            fit = powerlaw.Fit(sizes, discrete=True)
            R, p = fit.distribution_compare(
                "power_law", "exponential", normalized_ratio=True
            )

            # Ensure is_powerlaw is a standard Python boolean
            is_powerlaw = bool(R > 0 and p < 0.1)

            return {
                "alpha": float(fit.alpha) if fit.alpha is not None else None,
                "sigma_error": (
                    float(fit.sigma) if hasattr(fit, "sigma") else None
                ),  # Standard error of alpha
                "xmin": float(fit.xmin) if fit.xmin is not None else None,
                "is_powerlaw": is_powerlaw,
            }
        except Exception:
            return {"alpha": None, "sigma_error": None, "is_powerlaw": False}

    def find_outliers(self, n_neighbors: int = 5) -> Dict[str, float]:
        """
        Detect outliers using nearest neighbors approach.

        Args:
            n_neighbors: Number of neighbors to consider

        Returns:
            Dictionary mapping text indices to outlier scores
        """
        try:
            # Skip if we have too few samples
            if len(self.embeddings) < n_neighbors + 1:
                logger.warning("Not enough samples to detect outliers")
                return {}

            # Fit nearest neighbors
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(self.embeddings)

            # Get distances to nearest neighbors
            distances, _ = nn.kneighbors(self.embeddings)

            # Calculate outlier score as mean distance to neighbors
            outlier_scores = distances.mean(axis=1)

            # Create dictionary of outlier scores
            result = {}
            for i, score in enumerate(outlier_scores):
                result[i] = float(score)

            return result

        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return {}

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.

        The report includes:
        - basic_metrics: Contains clustering parameters (alpha, sigma) used as inputs
          to the clustering algorithms
        - powerlaw_params: Contains power law parameters (alpha, sigma_error) detected
          in the cluster size distribution. Note that this alpha is different from
          the clustering alpha parameter.

        Returns:
            Dictionary containing evaluation metrics
        """
        # Convert cluster assignments to regular Python list to avoid NumPy types
        cluster_assignments = [int(c) for c in self.cluster_assignments]

        # Create cluster size distribution using regular Python types
        cluster_size_distribution = {}
        for c in self.unique_clusters:
            cluster_size_distribution[str(c)] = cluster_assignments.count(c)

        report = {
            "basic_metrics": {
                "model_name": self.model_name,
                "num_texts": len(self.texts),
                "num_clusters": len(self.unique_clusters),
                "cluster_size_distribution": cluster_size_distribution,
                "alpha": self.alpha,
                "sigma": self.sigma,
            },
            "silhouette_score": self.calculate_silhouette_score(),
            "similarity_metrics": self.calculate_similarity_metrics(),
            "powerlaw_params": self.detect_powerlaw_distribution(),
            "outliers": self.find_outliers(),
            "cluster_assignments": cluster_assignments,
        }

        return report


def _sanitize_for_json(obj):
    """Convert NumPy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):  # type: ignore
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def _debug_json_error(report):
    """Debug JSON serialization errors by identifying problematic values."""
    for model_name, model_report in report.items():
        try:
            json.dumps(model_report, cls=NumpyEncoder)
        except TypeError:
            logger.error(f"Problem in model report: {model_name}")

            for key, value in model_report.items():
                try:
                    json.dumps({key: value}, cls=NumpyEncoder)
                except TypeError:
                    logger.error(f"Problem with key: {key}, value type: {type(value)}")


def _create_simplified_report(report):
    """Create a simplified version of the report with only basic metrics."""
    simplified_report = {}
    for model_name, model_report in report.items():
        simplified_report[model_name] = {
            "basic_metrics": model_report.get("basic_metrics", {}),
            "silhouette_score": model_report.get("silhouette_score", 0.0),
        }
    return simplified_report


def save_evaluation_report(
    report: Dict[str, Any], output_dir: str, filename: str = "evaluation_report.json"
) -> str:
    """
    Save the evaluation report to a JSON file.

    Args:
        report: Dictionary containing the evaluation report
        output_dir: Directory to save the report
        filename: Name of the output file

    Returns:
        Path to the saved report file
    """
    output_path = os.path.join(output_dir, filename)

    try:
        # Sanitize the report
        sanitized_report = _sanitize_for_json(report)

        with open(output_path, "w") as f:
            json.dump(sanitized_report, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Evaluation report saved to {output_path}")
        return output_path
    except TypeError as e:
        # If we still have serialization issues, log detailed information
        logger.error(f"JSON serialization error: {e}")

        # Debug the error
        _debug_json_error(report)

        # Save a simplified version
        simplified_report = _create_simplified_report(report)
        with open(output_path, "w") as f:
            json.dump(simplified_report, f, indent=2)

        logger.warning(
            f"Saved simplified report to {output_path} due to serialization issues"
        )
        return output_path
