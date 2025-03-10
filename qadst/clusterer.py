"""HDBSCAN-based QA dataset clustering implementation."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

from .base import BaseClusterer
from .utils import if_ok, is_numeric

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
    """

    def cluster_questions(self, qa_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Cluster questions based on semantic similarity using HDBSCAN.

        Args:
            qa_pairs: List of (question, answer) tuples

        Returns:
            Dict containing clustering results in the requested format

        Example:
            >>> from qadst.embeddings import get_embeddings_model, EmbeddingsProvider
            >>> # Create embeddings provider
            >>> model = get_embeddings_model("text-embedding-3-large")
            >>> provider = EmbeddingsProvider(model=model)
            >>> # Create clusterer with the provider
            >>> clusterer = HDBSCANQAClusterer(embeddings_provider=provider)
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
            >>> # Get the number of clusters
            >>> num_clusters = len(clusters['clusters'])
            >>> print(f"Number of clusters: {num_clusters}")
            Number of clusters: 2
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
        if not qa_pairs:
            return {"clusters": []}

        return self._perform_hdbscan_clustering(qa_pairs)

    def cluster_method(self) -> str:
        """Return the clustering method name.

        Returns:
            String identifier for the clustering method
        """
        return "hdbscan"

    def _calculate_min_cluster_size(self, total_questions: int) -> int:
        """Calculate the minimum cluster size based on the total number of questions.

        Uses logarithmic scaling to determine the appropriate minimum cluster size,
        which better represents how semantic clusters naturally form in QA datasets.
        This approach is supported by academic research showing that cluster size
        should scale sublinearly with dataset size.

        Args:
            total_questions: Total number of questions in the dataset

        Returns:
            Minimum cluster size for HDBSCAN

        Example:
            >>> clusterer = HDBSCANQAClusterer("text-embedding-3-large")
            >>> # Small dataset (30 questions)
            >>> clusterer._calculate_min_cluster_size(30)
            11
            >>> # Medium dataset (150 questions)
            >>> clusterer._calculate_min_cluster_size(150)
            25
            >>> # Large dataset (500 questions)
            >>> clusterer._calculate_min_cluster_size(500)
            38
            >>> # Very large dataset (3000 questions)
            >>> clusterer._calculate_min_cluster_size(3000)
            64
        """
        if self.min_cluster_size is not None:
            return self.min_cluster_size

        base_size = max(3, int(np.log(total_questions) ** 2))

        # Cap at 100 for very large datasets to prevent overly large minimum clusters
        return min(base_size, 100)

    def _perform_hdbscan_clustering(
        self, qa_pairs: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """Perform HDBSCAN clustering on the given QA pairs.

        Args:
            qa_pairs: List of (question, answer) tuples

        Returns:
            Dict containing clustering results
        """
        if not qa_pairs:
            return {"clusters": []}

        questions = [q for q, _ in qa_pairs]

        # Use cached embeddings if available
        questions_hash = self._calculate_deterministic_hash(questions)
        model_name = self.embeddings_provider.get_model_name()
        cache_key = f"cluster_{model_name}_{questions_hash}"
        question_embeddings = self.embeddings_provider.get_embeddings(
            questions, cache_key
        )

        # Convert to numpy array for HDBSCAN
        embeddings_array = np.array(question_embeddings)

        total_questions = len(questions)

        # Use provided min_cluster_size or calculate it
        if self.min_cluster_size:
            min_cluster_size = self.min_cluster_size
        else:
            min_cluster_size = self._calculate_min_cluster_size(total_questions)

        # Use provided min_samples or default to 5
        min_samples = self.min_samples or 5

        # Use provided cluster_selection_epsilon or default to 0.3
        cluster_selection_epsilon = self.cluster_selection_epsilon or 0.3

        cluster_selection_method = self.cluster_selection_method or "eom"

        logger.info(
            f"Clustering {total_questions} questions with HDBSCAN"
            f"(min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}, "
            f"cluster_selection_epsilon={cluster_selection_epsilon}, "
            f"cluster_selection_method={cluster_selection_method})"
        )

        hdbscan = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
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

        # Process noise points if there are enough of them and keep_noise is False
        if (
            "-1" in clusters
            and len(clusters["-1"]["questions"]) > min_cluster_size * 2
            and not self.keep_noise
        ):
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
        question_embeddings = self.embeddings_provider.get_embeddings(questions)
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

    def _identify_large_clusters(
        self, clusters: Dict[str, Dict], max_cluster_size: int
    ) -> Dict[str, Dict]:
        """Identify clusters that exceed the maximum size threshold.

        Args:
            clusters: Dict of cluster ID to cluster data
            max_cluster_size: Maximum allowed cluster size

        Returns:
            Dict of large cluster ID to cluster data
        """
        large_clusters = {}
        for cluster_id, cluster_data in clusters.items():
            if len(cluster_data["questions"]) > max_cluster_size:
                large_clusters[cluster_id] = cluster_data

        if large_clusters:
            logger.info(f"Found {len(large_clusters)} large clusters to split")
        else:
            logger.debug("No large clusters found")

        return large_clusters

    def _get_recursive_hdbscan_params(self, cluster_size: int) -> Tuple[int, float]:
        """Calculate parameters for recursive HDBSCAN on a large cluster.

        Args:
            cluster_size: Number of questions in the cluster

        Returns:
            Tuple of (min_cluster_size, epsilon)
        """
        # Use a smaller min_cluster_size to encourage more fine-grained clusters
        recursive_min_cluster_size = max(
            int(np.log(cluster_size) ** 1.5),  # More aggressive scaling
            5,  # Minimum of 5 to avoid tiny clusters
        )

        # Use a smaller epsilon to be more strict about cluster boundaries
        recursive_epsilon = 0.2  # Tighter than the default 0.3

        logger.debug(
            "Recursive HDBSCAN parameters: "
            f"min_cluster_size={recursive_min_cluster_size}, "
            f"epsilon={recursive_epsilon}"
        )

        return recursive_min_cluster_size, recursive_epsilon

    def _apply_recursive_clustering(
        self, questions: List[str], embeddings_array: np.ndarray, cluster_id: str
    ) -> Tuple[np.ndarray, int]:
        """Apply recursive clustering to a large cluster.

        First tries HDBSCAN with stricter parameters, falls back to K-means if needed.

        Args:
            questions: List of questions in the cluster
            embeddings_array: Array of embeddings for the questions
            cluster_id: ID of the cluster being processed

        Returns:
            Tuple of (subcluster_labels, num_subclusters)
        """
        # Get parameters for recursive HDBSCAN
        cluster_size = len(questions)
        min_cluster_size, epsilon = self._get_recursive_hdbscan_params(cluster_size)

        cluster_selection_method = self.cluster_selection_method or "eom"

        # Apply HDBSCAN with stricter parameters
        # Use the same cluster selection method as the main clustering
        hdbscan = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=3,  # Slightly lower than default to allow smaller clusters
            cluster_selection_epsilon=epsilon,
            cluster_selection_method=cluster_selection_method,
        )

        subcluster_labels = hdbscan.fit_predict(embeddings_array)

        # Count the number of subclusters (excluding noise)
        num_subclusters = len(set(subcluster_labels))
        if -1 in subcluster_labels:
            num_subclusters -= 1

        # If HDBSCAN couldn't split the cluster effectively, fall back to K-means
        if num_subclusters <= 1:
            logger.info(
                f"Recursive HDBSCAN couldn't split cluster {cluster_id} effectively. "
                "Falling back to K-means."
            )

            # Apply K-means with adaptive number of clusters
            k = self._calculate_kmeans_clusters(len(questions))
            kmeans = KMeans(n_clusters=k, random_state=42)
            subcluster_labels = kmeans.fit_predict(embeddings_array)
            num_subclusters = k

        logger.info(f"Split cluster {cluster_id} into {num_subclusters} subclusters")
        return subcluster_labels, num_subclusters

    def _calculate_kmeans_clusters(self, num_questions: int) -> int:
        """Calculate the appropriate number of K-means clusters.

        Args:
            num_questions: Number of questions in the cluster

        Returns:
            Number of clusters to use for K-means
        """
        return min(
            # At least 2, aim for ~30 questions per cluster
            max(2, int(num_questions / 30)),
            # Cap at 10 subclusters to avoid excessive fragmentation
            10,
        )

    def _create_subclusters(
        self,
        cluster_id: str,
        questions: List[str],
        qa_pairs: List[Dict],
        subcluster_labels: np.ndarray,
    ) -> Dict[str, Dict]:
        """Create subclusters from the clustering results.

        Args:
            cluster_id: ID of the original cluster
            questions: List of questions in the cluster
            qa_pairs: List of QA pairs in the cluster
            subcluster_labels: Array of subcluster labels

        Returns:
            Dict of subcluster ID to subcluster data
        """
        subclusters = {}
        for i, label in enumerate(subcluster_labels):
            if label == -1:  # Skip noise points
                continue

            label_key = f"{cluster_id}.{label}"
            if label_key not in subclusters:
                subclusters[label_key] = {"questions": [], "qa_pairs": []}

            subclusters[label_key]["questions"].append(questions[i])
            subclusters[label_key]["qa_pairs"].append(qa_pairs[i])

        return subclusters

    def _handle_large_clusters(
        self, clusters: Dict[str, Dict], total_questions: int
    ) -> Dict[str, Dict]:
        """Handle large clusters by recursively applying HDBSCAN with stricter
        parameters.

        Large clusters often contain diverse content that should be further subdivided.
        Instead of using K-means (which doesn't respect the density-based nature of the
        original clustering), this method recursively applies HDBSCAN with stricter
        parameters to maintain consistency with the main algorithm.

        Args:
            clusters: Dict of cluster ID to cluster data
            total_questions: Total number of questions in the dataset

        Returns:
            Dict of cluster ID to cluster data with large clusters split
        """
        # Calculate the maximum cluster size threshold (20% of total questions)
        max_cluster_size = max(int(total_questions * 0.2), 50)
        logger.debug(f"Maximum cluster size threshold: {max_cluster_size}")

        # Identify large clusters
        large_clusters = self._identify_large_clusters(clusters, max_cluster_size)

        if not large_clusters:
            return clusters

        # Process each large cluster
        for cluster_id, cluster_data in large_clusters.items():
            # Skip the noise cluster if keep_noise is True
            # Check if it's "-1" or can be converted to -1
            is_noise_cluster = cluster_id == "-1" or (
                is_numeric(cluster_id) and if_ok(int, cluster_id) == -1
            )

            if is_noise_cluster and self.keep_noise:
                logger.info("Skipping noise cluster splitting as keep_noise=True")
                continue

            logger.info(
                f"Splitting large cluster {cluster_id} with "
                f"{len(cluster_data['questions'])} questions"
            )

            # Extract questions and embeddings for this cluster
            questions = cluster_data["questions"]
            qa_pairs = cluster_data["qa_pairs"]

            # Get embeddings for the questions
            question_embeddings = self.embeddings_provider.get_embeddings(questions)
            embeddings_array = np.array(question_embeddings)

            # Apply recursive clustering
            subcluster_labels, num_subclusters = self._apply_recursive_clustering(
                questions, embeddings_array, cluster_id
            )

            # Create subclusters
            subclusters = self._create_subclusters(
                cluster_id, questions, qa_pairs, subcluster_labels
            )

            # Remove the original large cluster
            del clusters[cluster_id]

            # Add the subclusters
            for subcluster_id, subcluster_data in subclusters.items():
                clusters[subcluster_id] = subcluster_data

        return clusters

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
        next_cluster_id = 1  # Start regular clusters from ID 1

        for cluster_id, cluster_data in clusters.items():
            if not cluster_data["qa_pairs"]:
                continue

            sources = cluster_data["qa_pairs"]

            # HDBSCAN uses -1 internally to represent noise points.
            # Check if it's "-1" or can be converted to -1.
            is_noise_cluster = cluster_id == "-1" or (
                is_numeric(cluster_id) and if_ok(int, cluster_id) == -1
            )

            if is_noise_cluster and self.keep_noise:
                id = 0  # Use 0 for noise cluster
                representative = []  # No representative for noise
            else:
                id = next_cluster_id
                representative = [
                    {
                        "question": cluster_data["qa_pairs"][0]["question"],
                        "answer": cluster_data["qa_pairs"][0]["answer"],
                    }
                ]

            formatted_clusters.append(
                {
                    "id": id,
                    "representative": representative,
                    "source": sources,
                }
            )
            next_cluster_id += 1

        # Sort clusters by ID
        formatted_clusters.sort(key=lambda x: x["id"])

        return {"clusters": formatted_clusters}
