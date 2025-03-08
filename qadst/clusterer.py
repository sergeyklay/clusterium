import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

from .base import BaseClusterer

logger = logging.getLogger(__name__)


class HDBSCANQAClusterer(BaseClusterer):
    def __init__(
        self,
        embedding_model_name: str,
        llm_model_name: Optional[str] = None,
        output_dir: str = "./output",
        filter_enabled: bool = True,
    ):
        """Initialize the HDBSCAN clusterer.

        Args:
            embedding_model_name: Name of the embedding model to use
            output_dir: Directory to save output files
            llm_model_name: Optional name of the LLM model to use for filtering
               and labeling
            filter_enabled: Whether to enable filtering of engineering questions
        """
        super().__init__(
            embedding_model_name=embedding_model_name,
            output_dir=output_dir,
            llm_model_name=llm_model_name,
            filter_enabled=filter_enabled,
        )

    def cluster_questions(self, qa_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Cluster questions based on semantic similarity using HDBSCAN.

        Args:
            qa_pairs: List of (question, answer) tuples

        Returns:
            Dict containing clustering results in the requested format
        """
        return self._perform_hdbscan_clustering(qa_pairs)

    def cluster_method(self) -> str:
        """Return the name of the clustering method."""
        return "hdbscan"

    def _calculate_min_cluster_size(self, total_questions: int) -> int:
        """Calculate the minimum cluster size based on the total number of questions.

        Args:
            total_questions: Total number of questions

        Returns:
            Minimum cluster size
        """
        # For small datasets, we want smaller clusters
        if total_questions < 50:
            return max(3, total_questions // 15)
        # For medium datasets, we want larger clusters
        elif total_questions < 200:
            return max(5, total_questions // 20)
        # For large datasets, we want even larger clusters
        else:
            return max(8, total_questions // 25)

    def _perform_hdbscan_clustering(
        self, qa_pairs: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """Perform clustering using HDBSCAN.

        This method performs clustering using HDBSCAN, which automatically
        determines the optimal number of clusters.

        Args:
            qa_pairs: List of (question, answer) tuples

        Returns:
            Dict containing clustering results
        """
        if not qa_pairs:
            return {"clusters": []}

        questions = [q for q, _ in qa_pairs]
        question_embeddings = self.embeddings_model.embed_documents(questions)
        embeddings_array = np.array(question_embeddings)

        total_questions = len(questions)
        min_cluster_size = self._calculate_min_cluster_size(total_questions)

        logger.info(
            f"Clustering {total_questions} questions with HDBSCAN "
            f"(min_cluster_size={min_cluster_size})"
        )

        # Initialize HDBSCAN with appropriate parameters
        hdbscan = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=None,  # Use default (same as min_cluster_size)
            metric="euclidean",
            # Excess of Mass - usually gives better results
            cluster_selection_method="eom",
            # Small epsilon to merge very similar clusters
            cluster_selection_epsilon=0.1,
            alpha=1.0,
            algorithm="best",
            leaf_size=40,
        )

        # Fit and predict clusters
        cluster_labels = hdbscan.fit_predict(embeddings_array)

        # Count number of actual clusters (excluding noise points labeled as -1)
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logger.info(
            f"HDBSCAN found {num_clusters} clusters and "
            f"{np.sum(cluster_labels == -1)} noise points"
        )

        # Organize data into clusters
        clusters = {}
        for i, label in enumerate(cluster_labels):
            # Convert label to string for consistency with the rest of the code
            label_key = str(int(label))

            # Skip noise points (label -1) or create a special noise cluster if needed
            if label == -1:
                if "-1" not in clusters:
                    clusters["-1"] = {"questions": [], "qa_pairs": []}
                clusters["-1"]["questions"].append(questions[i])
                clusters["-1"]["qa_pairs"].append(
                    {"question": qa_pairs[i][0], "answer": qa_pairs[i][1]}
                )
                continue

            if label_key not in clusters:
                clusters[label_key] = {"questions": [], "qa_pairs": []}

            clusters[label_key]["questions"].append(questions[i])
            clusters[label_key]["qa_pairs"].append(
                {"question": qa_pairs[i][0], "answer": qa_pairs[i][1]}
            )

        # If we have too many noise points, try to cluster them separately
        if "-1" in clusters and len(clusters["-1"]["questions"]) > min_cluster_size * 2:
            noise_qa_pairs = [
                (q["question"], q["answer"]) for q in clusters["-1"]["qa_pairs"]
            ]
            noise_clusters = self._cluster_noise_points(
                noise_qa_pairs, min_cluster_size
            )

            # Remove the original noise cluster
            del clusters["-1"]

            # Add the new noise subclusters with unique IDs
            if clusters:
                max_cluster_id = max([int(k) for k in clusters.keys()])
            else:
                max_cluster_id = -1

            for i, (_, subcluster) in enumerate(noise_clusters.items()):
                new_id = str(max_cluster_id + i + 1)
                clusters[new_id] = subcluster

        # Handle any remaining large clusters
        final_clusters = self._handle_large_clusters(clusters, total_questions)
        return self._format_clusters(final_clusters)

    def _cluster_noise_points(
        self, noise_qa_pairs: List[Tuple[str, str]], min_cluster_size: int
    ) -> Dict[str, Dict]:
        """Attempt to cluster noise points using K-means with a small number of clusters.

        Args:
            noise_qa_pairs: List of (question, answer) tuples for noise points
            min_cluster_size: Minimum cluster size to use

        Returns:
            Dict of subclusters for noise points
        """
        if not noise_qa_pairs:
            return {}

        questions = [q for q, _ in noise_qa_pairs]
        question_embeddings = self.embeddings_model.embed_documents(questions)
        embeddings_array = np.array(question_embeddings)

        total_noise_points = len(questions)
        n_clusters = max(2, total_noise_points // (min_cluster_size * 2))

        logger.info(
            f"Attempting to cluster {total_noise_points} noise points into "
            f"{n_clusters} subclusters"
        )

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)

        subclusters = {}
        for i, label in enumerate(cluster_labels):
            label_key = str(int(label))

            if label_key not in subclusters:
                subclusters[label_key] = {"questions": [], "qa_pairs": []}

            subclusters[label_key]["questions"].append(questions[i])
            subclusters[label_key]["qa_pairs"].append(
                {"question": noise_qa_pairs[i][0], "answer": noise_qa_pairs[i][1]}
            )

        return subclusters

    def _handle_large_clusters(
        self, clusters: Dict[str, Dict], total_questions: int
    ) -> Dict[str, Dict]:
        """Split large clusters into smaller subclusters.

        Args:
            clusters: Initial clustering result
            total_questions: Total number of questions

        Returns:
            Dict with refined clustering
        """
        large_cluster_threshold = max(30, total_questions // 5)
        final_clusters = {}
        cluster_counter = 0

        for label, cluster_data in clusters.items():
            if len(cluster_data["questions"]) > large_cluster_threshold:
                logger.info(
                    f"Splitting large cluster {label} with "
                    f"{len(cluster_data['questions'])} questions"
                )

                cluster_embeddings = self.embeddings_model.embed_documents(
                    cluster_data["questions"]
                )
                cluster_embeddings_array = np.array(cluster_embeddings)
                num_subclusters = max(2, len(cluster_data["questions"]) // 20)

                subcluster_kmeans = KMeans(
                    n_clusters=num_subclusters, random_state=42, n_init=10
                )
                subcluster_labels = subcluster_kmeans.fit_predict(
                    cluster_embeddings_array
                )

                subclusters = {}
                for i, sublabel in enumerate(subcluster_labels):
                    sublabel_key = int(sublabel)
                    if sublabel_key not in subclusters:
                        subclusters[sublabel_key] = {"questions": [], "qa_pairs": []}
                    subclusters[sublabel_key]["questions"].append(
                        cluster_data["questions"][i]
                    )
                    subclusters[sublabel_key]["qa_pairs"].append(
                        cluster_data["qa_pairs"][i]
                    )

                for _, subcluster_data in subclusters.items():
                    final_clusters[str(cluster_counter)] = subcluster_data
                    cluster_counter += 1
            else:
                final_clusters[str(cluster_counter)] = cluster_data
                cluster_counter += 1

        return final_clusters

    def _format_clusters(self, clusters: Dict[str, Dict]) -> Dict[str, List]:
        """Format clusters in the requested JSON structure.

        Args:
            clusters: Dictionary of clusters

        Returns:
            Dict with clusters in the requested format
        """
        formatted_clusters = []

        for cluster_id, cluster_data in clusters.items():
            if not cluster_data["qa_pairs"]:
                continue

            representative = [
                {
                    "question": cluster_data["qa_pairs"][0]["question"],
                    "answer": cluster_data["qa_pairs"][0]["answer"],
                }
            ]

            sources = cluster_data["qa_pairs"]

            formatted_clusters.append(
                {
                    "id": int(cluster_id) + 1,  # Make IDs 1-based
                    "representative": representative,
                    "source": sources,
                }
            )

        return {"clusters": formatted_clusters}
