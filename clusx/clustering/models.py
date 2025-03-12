"""
Clustering models for text data using Dirichlet Process and Pitman-Yor Process.
"""

from collections.abc import Callable
from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine
from scipy.special import logsumexp
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm import tqdm

from clusx.logging import get_logger

logger = get_logger(__name__)


class DirichletProcess:
    """
    Dirichlet Process clustering implementation for text data.

    This implementation uses a Chinese Restaurant Process (CRP) formulation with
    proper Bayesian inference to cluster text data. It combines the CRP prior with
    a likelihood model based on multivariate Gaussian distributions in the embedding
    space.

    Attributes:
        alpha (float): Concentration parameter for new cluster creation.
            Higher values lead to more clusters.
        clusters (list[int]): List of cluster assignments for each processed text.
        cluster_params (dict): Dictionary of cluster parameters for each cluster.
            Contains 'mean' (centroid) and 'count' (number of points).
        model: Sentence transformer model used for text embeddings.
        random_state (numpy.random.RandomState): Random state for reproducibility.
    """

    def __init__(
        self,
        alpha: float,
        base_measure: Optional[dict] = None,
        similarity_metric: Optional[Callable[[Tensor, Tensor], float]] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize a Dirichlet Process clustering model.

        Args:
            alpha (float): Concentration parameter for new cluster creation.
                Higher values lead to more clusters.
            base_measure (Optional[dict]): Base measure parameters for the DP.
                Should contain 'variance' key for the likelihood model.
            similarity_metric (Optional[Callable]): Function to compute similarity
                between embeddings. If None, uses cosine_similarity.
            random_state (Optional[int]): Random seed for reproducibility.
        """
        self.alpha = alpha
        # Ensure base_measure has a variance value
        if base_measure is None:
            self.base_measure = {"variance": 0.1}
        else:
            self.base_measure = base_measure
        self.clusters: list[int] = []
        self.cluster_params: dict[int, dict] = {}
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.similarity_metric = (
            similarity_metric if similarity_metric else self.cosine_similarity
        )
        self.embedding_dim = None  # Will be set on first embedding

        # For reproducibility
        self.random_state = np.random.RandomState(random_state)

        # For tracking processed texts and their embeddings
        self.text_embeddings: dict[str, Tensor] = {}

    def get_embedding(self, text: str) -> Tensor:
        """
        Get the embedding for a text.

        Args:
            text (str): The text to embed.

        Returns:
            torch.Tensor: The embedding vector for the text.
        """
        # Check if already computed in this session
        if text in self.text_embeddings:
            return self.text_embeddings[text]

        embedding = self.model.encode(text)

        # Set embedding dimension if not set
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)

        # Store in session cache
        self.text_embeddings[text] = embedding

        return embedding

    def cosine_similarity(self, embedding1: Tensor, embedding2: Tensor) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1 (torch.Tensor): First embedding.
            embedding2 (torch.Tensor): Second embedding.

        Returns:
            float: Similarity score between 0 and 1, where 1 means identical.
        """
        similarity = 1 - cosine(embedding1, embedding2)
        return max(0.0, similarity)

    def log_likelihood(self, embedding: Tensor, cluster_id: int) -> float:
        """
        Calculate log likelihood of an embedding under a cluster's distribution.

        This implements a multivariate Gaussian likelihood in the embedding space.

        Args:
            embedding (torch.Tensor): The embedding to evaluate.
            cluster_id (int): The cluster ID.

        Returns:
            float: Log likelihood of the embedding under the cluster distribution.
        """
        if cluster_id not in self.cluster_params:
            # For new clusters, use the base measure
            return self._log_likelihood_base_measure(embedding)

        # Get cluster parameters
        cluster_mean = self.cluster_params[cluster_id]["mean"]
        variance = self.base_measure["variance"]

        # Calculate squared Mahalanobis distance (simplified for diagonal covariance)
        squared_dist = np.sum((embedding - cluster_mean) ** 2) / variance

        # Log likelihood of multivariate Gaussian
        if self.embedding_dim is not None:
            dim = float(self.embedding_dim)
        else:
            dim = float(len(embedding))

        log_likelihood = -0.5 * dim * np.log(2 * np.pi * variance) - 0.5 * squared_dist

        return log_likelihood

    def _log_likelihood_base_measure(self, embedding: Tensor) -> float:
        """
        Calculate log likelihood of an embedding under the base measure.

        Args:
            embedding (torch.Tensor): The embedding to evaluate.

        Returns:
            float: Log likelihood of the embedding under the base measure.
        """
        # For the base measure, we use a wider variance
        # Ensure we have a valid variance value
        variance = float(self.base_measure.get("variance", 0.1)) * 10.0

        # For new clusters, we center at the embedding itself
        if self.embedding_dim is not None:
            dim = float(self.embedding_dim)
        else:
            # Fallback if embedding_dim is not set
            dim = float(len(embedding))

        log_likelihood = -0.5 * dim * np.log(2 * np.pi * variance)

        return log_likelihood

    def log_crp_prior(self, cluster_id: int, total_points: int) -> float:
        """
        Calculate log probability of the Chinese Restaurant Process prior.

        Args:
            cluster_id (int): The cluster ID.
            total_points (int): Total number of points assigned so far.

        Returns:
            float: Log probability of the cluster under the CRP prior.
        """
        if cluster_id in self.cluster_params:
            # Existing cluster
            cluster_size = self.cluster_params[cluster_id]["count"]
            return np.log(cluster_size / (self.alpha + total_points))
        else:
            # New cluster
            return np.log(self.alpha / (self.alpha + total_points))

    def assign_cluster(self, text: str) -> int:
        """
        Assign a text to a cluster using proper Bayesian inference.

        This method computes log probabilities for assigning the text to each existing
        cluster or creating a new cluster. The probabilities combine:
        1. The CRP prior (Chinese Restaurant Process)
        2. The likelihood of the text embedding under each cluster's distribution

        Args:
            text (str): The text to assign to a cluster.

        Returns:
            int: The assigned cluster ID.

        Side effects:
            Updates self.clusters with the cluster assignment for this text.
            Updates self.cluster_params with the updated cluster parameters.
        """
        # Get embedding for the text
        embedding = self.get_embedding(text)

        # Calculate total points assigned so far
        total_points = len(self.clusters)

        # For the first point, always create a new cluster
        if total_points == 0:
            new_cluster_id = 0
            self.clusters.append(new_cluster_id)

            # Convert embedding to numpy array for storage
            if hasattr(embedding, "clone"):
                # PyTorch tensor
                emb_array = embedding.clone().detach().numpy()
            else:
                # Already numpy or other array-like
                emb_array = np.array(embedding)

            self.cluster_params[new_cluster_id] = {"mean": emb_array, "count": 1}

            return new_cluster_id

        # Calculate log probabilities for each existing cluster and a potential
        # new cluster
        log_probs = []
        cluster_ids = list(self.cluster_params.keys())

        # Consider existing clusters
        for cluster_id in cluster_ids:
            # Combine log prior and log likelihood
            log_prior = self.log_crp_prior(cluster_id, total_points)
            log_like = self.log_likelihood(embedding, cluster_id)
            log_probs.append(log_prior + log_like)

        # Consider creating a new cluster
        new_cluster_id = max(cluster_ids) + 1 if cluster_ids else 0
        log_prior_new = self.log_crp_prior(new_cluster_id, total_points)

        # For a new cluster, the likelihood is high because it would be centered
        # at the embedding. We use a fixed high value to encourage new cluster
        # formation
        if self.embedding_dim is not None:
            dim = float(self.embedding_dim)
        else:
            dim = float(len(embedding))

        # Base measure likelihood with a bonus to encourage new clusters
        variance = float(self.base_measure.get("variance", 0.1))
        log_like_new = -0.5 * dim * np.log(2 * np.pi * variance)  # Normalization term
        # No squared distance term because the cluster would be
        # centered at the embedding

        log_probs.append(log_prior_new + log_like_new)

        # Convert to probabilities and normalize
        log_probs = np.array(log_probs)
        log_probs -= logsumexp(log_probs)  # Normalize in log space
        probs = np.exp(log_probs)

        # Sample from the probability distribution
        all_cluster_ids = cluster_ids + [new_cluster_id]
        choice = self.random_state.choice(len(all_cluster_ids), p=probs)
        chosen_cluster_id = all_cluster_ids[choice]

        # Update cluster assignments and parameters
        self.clusters.append(chosen_cluster_id)

        if chosen_cluster_id not in self.cluster_params:
            # Create a new cluster
            # Convert embedding to numpy array for storage
            if hasattr(embedding, "clone"):
                # PyTorch tensor
                emb_array = embedding.clone().detach().numpy()
            else:
                # Already numpy or other array-like
                emb_array = np.array(embedding)

            self.cluster_params[chosen_cluster_id] = {"mean": emb_array, "count": 1}
        else:
            # Update existing cluster
            current_mean = self.cluster_params[chosen_cluster_id]["mean"]
            current_count = self.cluster_params[chosen_cluster_id]["count"]

            # Convert embedding to numpy for calculation
            if hasattr(embedding, "numpy"):
                emb_array = embedding.numpy()
            else:
                emb_array = np.array(embedding)

            # Update mean using online update formula
            new_mean = (current_mean * current_count + emb_array) / (current_count + 1)

            self.cluster_params[chosen_cluster_id]["mean"] = new_mean
            self.cluster_params[chosen_cluster_id]["count"] += 1

        return chosen_cluster_id

    def fit(self, texts: list[str]) -> tuple[list[int], dict]:
        """
        Train the Dirichlet Process model on the given text data.

        This method processes each text in the input list, assigning it to a cluster
        using Bayesian inference with the Chinese Restaurant Process prior.

        Args:
            texts (list[str]): List of text strings to cluster.

        Returns:
            tuple[list[int], dict]: A tuple containing:
                - List of cluster assignments for each text
                - Dictionary of cluster parameters
        """
        logger.info(f"Processing {len(texts)} texts with DirichletProcess...")

        # Reset state for a fresh run
        self.clusters = []
        self.cluster_params = {}

        for text in tqdm(texts, desc="Clustering"):
            self.assign_cluster(text)

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
    Process.

    Attributes:
        alpha (float): Concentration parameter inherited from DirichletProcess.
        sigma (float): Discount parameter controlling power-law behavior.
            Should be in range [0, 1). Higher values create more heavy-tailed
            distributions.
        clusters (list[int]): List of cluster assignments for each processed text.
        cluster_params (dict): Dictionary of cluster parameters for each cluster.
        model: Sentence transformer model used for text embeddings.
    """

    def __init__(
        self,
        alpha: float,
        sigma: float,
        base_measure: Optional[dict] = None,
        similarity_metric: Optional[Callable[[Tensor, Tensor], float]] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize a Pitman-Yor Process clustering model.

        Args:
            alpha (float): Concentration parameter that controls the propensity to
                create new clusters. Higher values lead to more clusters.
            sigma (float): Discount parameter controlling power-law behavior.
                Should be in range [0, 1). Higher values create more heavy-tailed
                distributions.
            base_measure (Optional[dict]): Base measure parameters for the PYP.
                Should contain 'variance' key for the likelihood model.
            similarity_metric (Optional[Callable]): Function to compute similarity
                between embeddings. If None, uses cosine_similarity.
            random_state (Optional[int]): Random seed for reproducibility.
        """
        super().__init__(alpha, base_measure, similarity_metric, random_state)

        # Validate sigma is in [0, 1)
        if not (0 <= sigma < 1):
            raise ValueError(f"Discount parameter sigma must be in [0, 1), got {sigma}")

        self.sigma = sigma

    def log_pyp_prior(self, cluster_id: int, total_points: int) -> float:
        """
        Calculate log probability of the Pitman-Yor Process prior.

        Args:
            cluster_id (int): The cluster ID.
            total_points (int): Total number of points assigned so far.

        Returns:
            float: Log probability of the cluster under the PYP prior.
        """
        if cluster_id in self.cluster_params:
            # Existing cluster
            cluster_size = self.cluster_params[cluster_id]["count"]
            return np.log(
                max(0, cluster_size - self.sigma) / (self.alpha + total_points)
            )
        else:
            # New cluster
            num_tables = len(self.cluster_params)
            return np.log(
                (self.alpha + self.sigma * num_tables) / (self.alpha + total_points)
            )

    def assign_cluster(self, text: str) -> int:
        """
        Assign a text to a cluster using proper Bayesian inference with PYP.

        This method computes log probabilities for assigning the text to each existing
        cluster or creating a new cluster. The probabilities combine:
        1. The PYP prior (Pitman-Yor Process)
        2. The likelihood of the text embedding under each cluster's distribution

        Args:
            text (str): The text to assign to a cluster.

        Returns:
            int: The assigned cluster ID.

        Side effects:
            Updates self.clusters with the cluster assignment for this text.
            Updates self.cluster_params with the updated cluster parameters.
        """
        # Get embedding for the text
        embedding = self.get_embedding(text)

        # Calculate total points assigned so far
        total_points = len(self.clusters)

        # For the first point, always create a new cluster
        if total_points == 0:
            new_cluster_id = 0
            self.clusters.append(new_cluster_id)

            # Convert embedding to numpy array for storage
            if hasattr(embedding, "clone"):
                # PyTorch tensor
                emb_array = embedding.clone().detach().numpy()
            else:
                # Already numpy or other array-like
                emb_array = np.array(embedding)

            self.cluster_params[new_cluster_id] = {"mean": emb_array, "count": 1}

            return new_cluster_id

        # Calculate log probabilities for each existing cluster and a potential
        # new cluster
        log_probs = []
        cluster_ids = list(self.cluster_params.keys())

        # Consider existing clusters
        for cluster_id in cluster_ids:
            # Combine log prior and log likelihood
            log_prior = self.log_pyp_prior(cluster_id, total_points)
            log_like = self.log_likelihood(embedding, cluster_id)
            log_probs.append(log_prior + log_like)

        # Consider creating a new cluster
        new_cluster_id = max(cluster_ids) + 1 if cluster_ids else 0
        log_prior_new = self.log_pyp_prior(new_cluster_id, total_points)

        # For a new cluster, the likelihood is high because it would be centered
        # at the embedding. We use a fixed high value to encourage new cluster
        # formation
        if self.embedding_dim is not None:
            dim = float(self.embedding_dim)
        else:
            dim = float(len(embedding))

        # Base measure likelihood with a bonus to encourage new clusters
        variance = float(self.base_measure.get("variance", 0.1))
        log_like_new = -0.5 * dim * np.log(2 * np.pi * variance)  # Normalization term
        # No squared distance term because the cluster would be
        # centered at the embedding

        log_probs.append(log_prior_new + log_like_new)

        # Convert to probabilities and normalize
        log_probs = np.array(log_probs)
        log_probs -= logsumexp(log_probs)  # Normalize in log space
        probs = np.exp(log_probs)

        # Sample from the probability distribution
        all_cluster_ids = cluster_ids + [new_cluster_id]
        choice = self.random_state.choice(len(all_cluster_ids), p=probs)
        chosen_cluster_id = all_cluster_ids[choice]

        # Update cluster assignments and parameters
        self.clusters.append(chosen_cluster_id)

        if chosen_cluster_id not in self.cluster_params:
            # Create a new cluster
            # Convert embedding to numpy array for storage
            if hasattr(embedding, "clone"):
                # PyTorch tensor
                emb_array = embedding.clone().detach().numpy()
            else:
                # Already numpy or other array-like
                emb_array = np.array(embedding)

            self.cluster_params[chosen_cluster_id] = {"mean": emb_array, "count": 1}
        else:
            # Update existing cluster
            current_mean = self.cluster_params[chosen_cluster_id]["mean"]
            current_count = self.cluster_params[chosen_cluster_id]["count"]

            # Convert embedding to numpy for calculation
            if hasattr(embedding, "numpy"):
                emb_array = embedding.numpy()
            else:
                emb_array = np.array(embedding)

            # Update mean using online update formula
            new_mean = (current_mean * current_count + emb_array) / (current_count + 1)

            self.cluster_params[chosen_cluster_id]["mean"] = new_mean
            self.cluster_params[chosen_cluster_id]["count"] += 1

        return chosen_cluster_id

    def fit(self, texts: list[str]) -> tuple[list[int], dict]:
        """
        Train the Pitman-Yor Process model on the given text data.

        This method processes each text in the input list, assigning it to a cluster
        using Bayesian inference with the Pitman-Yor Process prior.

        Args:
            texts (list[str]): List of text strings to cluster.

        Returns:
            tuple[list[int], dict]: A tuple containing:
                - List of cluster assignments for each text
                - Dictionary of cluster parameters
        """
        logger.info(f"Processing {len(texts)} texts with PitmanYorProcess...")

        # Reset state for a fresh run
        self.clusters = []
        self.cluster_params = {}

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

        return self.clusters, self.cluster_params
