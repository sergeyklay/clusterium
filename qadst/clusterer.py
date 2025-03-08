import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

from .base import BaseClusterer

logger = logging.getLogger(__name__)


class HDBSCANQAClusterer(BaseClusterer):
    """Question-answer clustering implementation using HDBSCAN algorithm.

    This class implements a density-based clustering approach using HDBSCAN
    (Hierarchical Density-Based Spatial Clustering of Applications with Noise),
    which automatically determines the optimal number of clusters and identifies
    noise points. The implementation includes additional processing for noise points
    and large clusters.

    HDBSCAN is particularly effective for datasets with varying cluster densities
    and non-spherical cluster shapes. It can identify outliers as noise points.

    References:
        - Campello, R.J.G.B., Moulavi, D., Sander, J. (2013) Density-Based Clustering
          Based on Hierarchical Density Estimates
        - McInnes, L., Healy, J., Astels, S. (2017) HDBSCAN: Hierarchical density
          based clustering
    """

    def cluster_questions(self, qa_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Cluster questions based on semantic similarity using HDBSCAN.

        Args:
            qa_pairs: List of (question, answer) tuples

        Returns:
            Dict containing clustering results in the requested format

        Example:
            >>> clusterer = HDBSCANQAClusterer("text-embedding-3-large")
            >>> qa_pairs = [
            ...     ("How do I reset my password?", "Click 'Forgot Password'"),
            ...     ("How do I change my email?", "Go to account settings"),
            ...     ("What payment methods do you accept?", "Credit cards and PayPal"),
            ...     ("Can I pay with Bitcoin?", "Yes, we accept cryptocurrency")
            ... ]
            >>> clusters = clusterer.cluster_questions(qa_pairs)
            >>> # The result contains cluster information
            >>> print(clusters.keys())
            dict_keys(['clusters'])
            >>> # Questions are grouped by semantic similarity
            >>> for cluster in clusters['clusters']:
            ...     print(f"Cluster {cluster['id']}: {len(cluster['source'])} items")
            Cluster 1: 2 items
            Cluster 2: 2 items
            >>> # Each cluster has a representative question and source questions
            >>> print(clusters['clusters'][0].keys())
            dict_keys(['id', 'representative', 'source'])
            >>> # Representative is the canonical question for the cluster
            >>> print(clusters['clusters'][0]['representative'][0]['question'])
            How do I reset my password?
        """
        return self._perform_hdbscan_clustering(qa_pairs)

    def cluster_method(self) -> str:
        """Return the name of the clustering method.

        Returns:
            String identifier for the HDBSCAN clustering method
        """
        return "hdbscan"

    def _calculate_min_cluster_size(self, total_questions: int) -> int:
        """Calculate the minimum cluster size based on the total number of questions.

        Adaptively determines appropriate minimum cluster size based on dataset size:
        - Small datasets: smaller clusters (higher granularity)
        - Medium datasets: moderate clusters
        - Large datasets: larger clusters (lower granularity)

        Args:
            total_questions: Total number of questions in the dataset

        Returns:
            Minimum cluster size for HDBSCAN

        Example:
            >>> clusterer = HDBSCANQAClusterer("text-embedding-3-large")
            >>> # Small dataset (30 questions)
            >>> clusterer._calculate_min_cluster_size(30)
            3
            >>> # Medium dataset (150 questions)
            >>> clusterer._calculate_min_cluster_size(150)
            7
            >>> # Large dataset (500 questions)
            >>> clusterer._calculate_min_cluster_size(500)
            20
        """
        if total_questions < 50:
            return max(3, total_questions // 15)
        elif total_questions < 200:
            return max(5, total_questions // 20)
        else:
            return max(8, total_questions // 25)

    def _perform_hdbscan_clustering(
        self, qa_pairs: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """Perform clustering using HDBSCAN algorithm.

        This method:
        1. Computes embeddings for all questions
        2. Applies HDBSCAN clustering with adaptive parameters
        3. Processes noise points separately using K-means
        4. Handles large clusters by splitting them
        5. Formats the results into the required structure

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

        hdbscan = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=None,  # Use default (same as min_cluster_size)
            metric="euclidean",
            cluster_selection_method="eom",  # Excess of Mass - better results
            cluster_selection_epsilon=0.1,  # Small epsilon to merge similar clusters
            alpha=1.0,
            algorithm="best",
            leaf_size=40,
        )

        cluster_labels = hdbscan.fit_predict(embeddings_array)

        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logger.info(
            f"HDBSCAN found {num_clusters} clusters and "
            f"{np.sum(cluster_labels == -1)} noise points"
        )

        clusters = {}
        for i, label in enumerate(cluster_labels):
            label_key = str(int(label))

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

        # Process noise points if there are enough of them
        if "-1" in clusters and len(clusters["-1"]["questions"]) > min_cluster_size * 2:
            noise_qa_pairs = [
                (q["question"], q["answer"]) for q in clusters["-1"]["qa_pairs"]
            ]
            noise_clusters = self._cluster_noise_points(
                noise_qa_pairs, min_cluster_size
            )

            del clusters["-1"]

            if clusters:
                max_cluster_id = max([int(k) for k in clusters.keys()])
            else:
                max_cluster_id = -1

            for i, (_, subcluster) in enumerate(noise_clusters.items()):
                new_id = str(max_cluster_id + i + 1)
                clusters[new_id] = subcluster

        final_clusters = self._handle_large_clusters(clusters, total_questions)
        return self._format_clusters(final_clusters)

    def _cluster_noise_points(
        self, noise_qa_pairs: List[Tuple[str, str]], min_cluster_size: int
    ) -> Dict[str, Dict]:
        """Cluster noise points using K-means to recover potentially useful groups.

        HDBSCAN identifies outliers as noise points (label -1). This method attempts
        to find meaningful subgroups within these noise points using K-means clustering,
        which can help recover useful information from points that don't fit the
        density-based clustering criteria.

        Args:
            noise_qa_pairs: List of (question, answer) tuples for noise points
            min_cluster_size: Minimum cluster size to use for determining K

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
        """Split large clusters into smaller subclusters for better granularity.

        Very large clusters may contain distinct subgroups that HDBSCAN merged due to
        density connectivity. This method applies K-means to large clusters to split
        them into more manageable and potentially more coherent subclusters.

        Args:
            clusters: Initial clustering result
            total_questions: Total number of questions in the dataset

        Returns:
            Dict with refined clustering after splitting large clusters
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
        """Format clusters into the standardized output structure.

        Converts the internal cluster representation to the required JSON structure
        with cluster IDs, representative questions, and source questions.

        Args:
            clusters: Dictionary of clusters with questions and QA pairs

        Returns:
            Dict with clusters in the standardized format
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
