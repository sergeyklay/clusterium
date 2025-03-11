"""
Clustering models for text data using Dirichlet Process and Pitman-Yor Process.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm import tqdm

from qadst.clustering.cache import EmbeddingCache
from qadst.logging import get_logger

logger = get_logger(__name__)


class DirichletProcess:
    """
    Dirichlet Process clustering implementation for text data.

    This implementation uses a Chinese Restaurant Process (CRP) formulation combined
    with semantic similarity measures to cluster text data.

    The model assigns each text to either an existing cluster or creates a new cluster
    based on both the CRP probabilities and the semantic similarity between the text
    and existing cluster centroids.

    Attributes:
        alpha (float): Concentration parameter for new cluster creation.
            Higher values lead to more clusters.
        clusters (list[int]): List of cluster assignments for each processed text.
        cluster_params (list[Tensor]): List of cluster embeddings for each cluster.
        model: Sentence transformer model used for text embeddings.
        cache (EmbeddingCache): Optional cache for storing text embeddings.
    """

    def __init__(
        self,
        alpha: float,
        base_measure: Optional[Tensor] = None,
        similarity_metric: Optional[Callable[[str, Tensor], float]] = None,
        cache: Optional[EmbeddingCache] = None,
    ):
        """
        Initialize a Dirichlet Process clustering model.

        Args:
            alpha (float): Concentration parameter for new cluster creation.
                Higher values lead to more clusters.
            base_measure (Optional[Tensor]): Base measure for the Dirichlet Process.
                Currently not used in this implementation.
            similarity_metric (Optional[Callable]): Function to compute similarity
                between a text and cluster parameters. If None, uses bert_similarity.
            cache (Optional[EmbeddingCache]): Cache for storing text embeddings.
                Helps avoid redundant embedding computations.
        """
        self.alpha = alpha
        self.base_measure = base_measure
        self.clusters: list[int] = []
        self.cluster_params: list[Tensor] = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.similarity_metric = (
            similarity_metric if similarity_metric else self.bert_similarity
        )

        self.cache = cache
        if self.cache:
            self.cache.load_cache()

    def get_embedding(self, text: str) -> Tensor:
        """
        Get the embedding for a text, using cache if available.

        Args:
            text (str): The text to embed.

        Returns:
            Tensor: The embedding vector for the text.
        """
        # Try to get from cache first
        if self.cache and text in self.cache:
            embedding = self.cache.get(text)
            if embedding is not None:
                return embedding

        # Generate new embedding
        embedding = self.model.encode(text)

        # Store in cache if provider available
        if self.cache:
            self.cache.set(text, embedding)

        return embedding

    def save_embedding_cache(self):
        """
        Save the embedding cache to disk if a cache provider is available.
        This helps preserve embeddings between runs for faster processing.
        """
        if self.cache:
            self.cache.save_cache()

    def bert_similarity(self, text, cluster_param):
        """
        Calculate cosine similarity between a text and cluster parameters.

        Args:
            text (str): The text to compare.
            cluster_param (Tensor): The cluster parameters (embedding).

        Returns:
            float: Similarity score between 0 and 1, where 1 means identical.
        """
        text_embedding = self.get_embedding(text)
        cluster_embedding = cluster_param
        similarity = 1 - cosine(text_embedding, cluster_embedding)
        return max(0.0, similarity)

    def sample_new_cluster(self, text):
        """
        Sample parameters for a new cluster based on the given text.

        Args:
            text (str): The text to use as the basis for the new cluster.

        Returns:
            Tensor: Embedding to use as parameters for the new cluster.
        """
        return self.get_embedding(text)

    def assign_cluster(self, text):
        """
        Assign a text to a cluster using the Chinese Restaurant Process with similarity.

        This method computes probabilities for assigning the text to each existing
        cluster or creating a new cluster. The probabilities are based on:
        1. The number of texts already in each cluster (CRP prior)
        2. The similarity between the text and each cluster's parameters
        3. The concentration parameter alpha

        The method then samples from this probability distribution to make the
        assignment.

        Args:
            text (str): The text to assign to a cluster.

        Side effects:
            Updates self.clusters with the cluster assignment for this text.
            Updates self.cluster_params if a new cluster is created.
        """
        probs = []
        total_points = len(self.clusters)

        for i, params in enumerate(self.cluster_params):
            cluster_size = self.clusters.count(i)
            prob = (
                cluster_size / (self.alpha + total_points)
            ) * self.similarity_metric(text, params)
            probs.append(max(0.0, prob))

        new_cluster_prob = (self.alpha / (self.alpha + total_points)) * 1.0
        probs.append(new_cluster_prob)

        probs = np.array(probs)
        if probs.sum() <= 0:
            probs = np.ones(len(probs)) / len(probs)
        else:
            probs /= probs.sum()

        choice = np.random.choice(len(probs), p=probs)
        if choice == len(self.cluster_params):
            new_params = self.sample_new_cluster(text)
            self.cluster_params.append(new_params)
            self.clusters.append(len(self.cluster_params) - 1)
        else:
            self.clusters.append(choice)

    def fit(self, texts: List[str]) -> Tuple[List[int], List[Tensor]]:
        """
        Train the Dirichlet Process model on the given text data.

        This method processes each text in the input list, assigning it to a cluster
        using the Chinese Restaurant Process.

        Args:
            texts (List[str]): List of text strings to cluster.

        Returns:
            Tuple[List[int], List[Tensor]]: A tuple containing:
                - List of cluster assignments for each text
                - List of cluster parameters (embeddings)
        """
        logger.info(f"Processing {len(texts)} texts...")
        for text in tqdm(texts, desc="Clustering"):
            self.assign_cluster(text)

        self.save_embedding_cache()

        return self.clusters, self.cluster_params


class PitmanYorProcess(DirichletProcess):
    """
    Pitman-Yor Process clustering implementation for text data.

    The Pitman-Yor Process is a generalization of the Dirichlet Process that introduces
    a discount parameter (sigma) to control the power-law behavior of the cluster
    size distribution. It is particularly effective for modeling natural language
    phenomena that exhibit power-law distributions.

    This implementation extends the DirichletProcess class, adding the sigma parameter
    and modifying the cluster assignment probabilities according to the Pitman-Yor
    Process. It also includes optimizations for tracking cluster sizes.

    Attributes:
        alpha (float): Concentration parameter inherited from DirichletProcess.
        sigma (float): Discount parameter controlling power-law behavior.
            Should be in range [0, 1). Higher values create more heavy-tailed
            distributions.
        clusters (list[int]): List of cluster assignments for each processed text.
        cluster_params (list[Tensor]): List of cluster parameters (embeddings) for
            each cluster.
        cluster_sizes (dict[int, int]): Dictionary tracking the size of each cluster.
        model: Sentence transformer model used for text embeddings.
        cache (EmbeddingCache): Optional cache for storing text embeddings.
    """

    def __init__(
        self,
        alpha: float,
        sigma: float,
        base_measure: Optional[Tensor] = None,
        similarity_metric: Optional[Callable[[str, Tensor], float]] = None,
        cache: Optional[EmbeddingCache] = None,
    ):
        """
        Initialize a Pitman-Yor Process clustering model.

        Args:
            alpha (float): Concentration parameter that controls the propensity to
                create new clusters. Higher values lead to more clusters.
            sigma (float): Discount parameter controlling power-law behavior.
                Should be in range [0, 1). Higher values create more heavy-tailed
                distributions.
            base_measure (Optional[Tensor]): Base measure for the Pitman-Yor Process.
                Currently not used in this implementation.
            similarity_metric (Optional[Callable[[str, Tensor], float]]): Function to
                compute similarity between a text and cluster parameters.
                If None, uses bert_similarity.
            cache (Optional[EmbeddingCache]): Cache for storing text embeddings.
                Helps avoid redundant embedding computations.
        """
        super().__init__(alpha, base_measure, similarity_metric, cache)
        self.sigma = sigma
        # Keep track of cluster sizes for faster access
        self.cluster_sizes = {}

    def assign_cluster(self, text):
        """
        Assign a text to a cluster using the Pitman-Yor Process with similarity.

        This method extends the DirichletProcess assignment method by modifying the
        probability calculations to incorporate the discount parameter sigma. The
        Pitman-Yor probabilities favor a power-law distribution of cluster sizes,
        which is often more realistic for natural language data.

        The method computes probabilities for assigning the text to each existing
        cluster or creating a new cluster based on:
        1. The number of texts already in each cluster (PYP prior)
        2. The similarity between the text and each cluster's parameters
        3. The concentration parameter alpha and discount parameter sigma

        Args:
            text (str): The text to assign to a cluster.

        Side effects:
            Updates self.clusters with the cluster assignment for this text.
            Updates self.cluster_params if a new cluster is created.
            Updates self.cluster_sizes to track cluster populations.
        """
        probs = []
        total_points = len(self.clusters)

        # Pre-compute the embedding once
        text_embedding = self.get_embedding(text)

        # Update cluster sizes dictionary
        if not hasattr(self, "cluster_sizes") or self.cluster_sizes is None:
            self.cluster_sizes = {}
            for i in range(len(self.cluster_params)):
                self.cluster_sizes[i] = self.clusters.count(i)

        for i, params in enumerate(self.cluster_params):
            # Use the cached cluster size instead of counting each time
            cluster_size = self.cluster_sizes.get(i, 0)
            adjusted_size = max(self.sigma, cluster_size)

            # Calculate similarity directly with embeddings for speed
            similarity = 1 - cosine(text_embedding, params)
            similarity = max(0.0, similarity)

            prob = (
                (adjusted_size - self.sigma) / (self.alpha + total_points) * similarity
            )
            probs.append(max(0.0, prob))

        new_cluster_prob = (
            (self.alpha + self.sigma * len(self.cluster_params))
            / (self.alpha + total_points)
        ) * 1.0
        probs.append(new_cluster_prob)

        probs = np.array(probs)
        if probs.sum() <= 0:
            probs = np.ones(len(probs)) / len(probs)
        else:
            probs /= probs.sum()

        choice = np.random.choice(len(probs), p=probs)
        if choice == len(self.cluster_params):
            # Use the already computed embedding
            self.cluster_params.append(text_embedding)
            self.clusters.append(len(self.cluster_params) - 1)
            # Update cluster sizes
            self.cluster_sizes[len(self.cluster_params) - 1] = 1
        else:
            self.clusters.append(choice)
            # Update cluster sizes
            self.cluster_sizes[choice] = self.cluster_sizes.get(choice, 0) + 1

    def fit(self, texts: List[str]) -> Tuple[List[int], List[Tensor]]:
        """
        Train the Pitman-Yor Process model on the given text data.

        This is an optimized version of the fit method for PitmanYorProcess that
        processes texts with tracking of cluster sizes for better performance.

        Args:
            texts (List[str]): List of text strings to cluster.

        Returns:
            Tuple[List[int], List[Tensor]]: A tuple containing:
                - List of cluster assignments for each text
                - List of cluster parameters (embeddings)
        """
        logger.info(f"Processing {len(texts)} texts with optimized PitmanYorProcess...")

        # Initialize cluster sizes dictionary
        self.cluster_sizes = {}

        # Process texts in batches for better progress reporting
        batch_size = 100
        total_batches = (len(texts) - 1) // batch_size + 1
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            for text in tqdm(
                batch, desc=f"Clustering batch {batch_num}/{total_batches}"
            ):
                self.assign_cluster(text)

        self.save_embedding_cache()
        return self.clusters, self.cluster_params
