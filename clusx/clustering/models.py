"""
Clustering models for text data using Dirichlet Process and Pitman-Yor Process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cosine
from scipy.special import logsumexp
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal, Optional, Union

    import torch
    from numpy.typing import NDArray

    EmbeddingTensor = Union[torch.Tensor, NDArray[np.float32]]

from clusx.logging import get_logger
from clusx.utils import to_numpy

logger = get_logger(__name__)


# TODO: Get rid of assert statements, use type checking and raise errors
class DirichletProcess:
    """
    Dirichlet Process clustering implementation for text data using von Mises-Fisher distribution.

    This implementation uses a Chinese Restaurant Process (CRP) formulation with
    Bayesian inference to cluster text data. It combines the CRP prior with
    a likelihood model based on von Mises-Fisher distributions in the embedding
    space, which is particularly suitable for directional data like normalized
    text embeddings.

    The model uses a concentration parameter alpha to control the propensity to
    create new clusters, and a precision parameter kappa to control the
    concentration of points around cluster means in the von Mises-Fisher distribution.

    Attributes:
        alpha (float): Concentration parameter for new cluster creation.
            Higher values lead to more clusters.
        kappa (float): Precision parameter for the von Mises-Fisher distribution.
            Higher values lead to tighter, more concentrated clusters.
        model (SentenceTransformer): Sentence transformer model used for text
            embeddings.
        random_state (numpy.random.Generator): Random state for reproducibility.
        clusters (list[int]): List of cluster assignments for each processed text.
        cluster_params (dict): Dictionary of cluster parameters for each cluster.
            Contains 'mean' (centroid) and 'count' (number of points).
        global_mean (ndarray): Global mean of all document embeddings.
        next_id (int): Next available cluster ID.
        embeddings_ (ndarray): Document embeddings after fitting.
        labels_ (ndarray): Cluster assignments after fitting.
        text_embeddings (dict): Cache of text to embedding mappings.
        embedding_dim (Optional[int]): Dimension of the embedding vectors.
    """  # noqa: E501

    def __init__(
        self,
        alpha: float,
        kappa: float,
        model_name: Optional[str] = "all-MiniLM-L6-v2",
        random_state: Optional[int] = None,
    ):
        """
        Initialize a Dirichlet Process model with von Mises-Fisher likelihood.

        Args:
            alpha (float): Concentration parameter for new cluster creation.
                Higher values lead to more clusters.
            kappa (float): Precision parameter for the von Mises-Fisher distribution.
                Higher values lead to tighter, more concentrated clusters.
            model_name (Optional[str]): Name of the sentence transformer model to use.
                Defaults to "all-MiniLM-L6-v2".
            random_state (Optional[int]): Random seed for reproducibility.
                If None, then fresh, unpredictable entropy will be pulled from the OS.
        """
        self.alpha = alpha
        self.kappa = kappa
        self.model = SentenceTransformer(model_name or "all-MiniLM-L6-v2")

        # For reproducibility
        self.random_state = np.random.default_rng(seed=random_state)

        self.clusters = []
        self.cluster_params = {}
        self.global_mean = None
        self.next_id = 0
        self.embeddings_ = None
        self.labels_ = None

        # For tracking processed texts and their embeddings
        self.text_embeddings: dict[str, EmbeddingTensor] = {}
        self.embedding_dim: Optional[int] = None  # Will be set on first embedding

    def get_embedding(self, text: Union[str, list[str]]) -> EmbeddingTensor:
        """
        Get the embedding for a text or list of texts with caching.

        This method computes embeddings for text inputs using the sentence transformer
        model. It implements caching to avoid recomputing embeddings for previously
        seen texts. The method can handle both single text strings and lists of texts.

        Args:
            text (Union[str, list[str]]): Text or list of texts to embed.

        Returns:
            EmbeddingTensor: The normalized embedding vector(s) for the text.
                If input is a single string, returns a single embedding vector.
                If input is a list, returns an array of embedding vectors.
        """
        # Handle single text vs list
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        # Get embeddings (using cache when available)
        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, t in enumerate(texts):
            if t in self.text_embeddings:
                results.append(self.text_embeddings[t])
            else:
                uncached_texts.append(t)
                uncached_indices.append(i)

        # Compute embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts, show_progress_bar=False)

            # Update cache and results
            for i, embedding in zip(uncached_indices, new_embeddings):
                t = texts[i]
                normalized_embedding = self._normalize(embedding)
                self.text_embeddings[t] = normalized_embedding
                results.append(normalized_embedding)

        # Ensure results are in the original order
        results = [results[texts.index(t)] for t in texts]

        # Set embedding dimension if not set
        if self.embedding_dim is None and results:
            self.embedding_dim = len(results[0])

        # Return single embedding or list based on input
        return results[0] if is_single else np.array(results)

    def _normalize(self, embedding: EmbeddingTensor) -> EmbeddingTensor:
        """
        Normalize vector to unit length for use with von Mises-Fisher distribution.

        The von Mises-Fisher distribution is defined on the unit hypersphere, so
        all vectors must be normalized to unit length.

        Args:
            embedding (EmbeddingTensor): The embedding vector to normalize.

        Returns:
            EmbeddingTensor: The normalized embedding vector with unit length.
        """
        norm = np.linalg.norm(embedding)
        # Convert to numpy array to ensure division works properly
        embedding_np = to_numpy(embedding)
        return embedding_np / norm if norm > 0 else embedding_np

    def _log_likelihood_base(
        self, embedding: EmbeddingTensor, cluster_id: int
    ) -> float:
        """
        Calculate von Mises-Fisher log-likelihood for a document in a cluster.

        The von Mises-Fisher distribution is a probability distribution on the
        unit hypersphere. For unit vectors x and μ, the log-likelihood is proportional
        to κ·(x·μ), where κ is the concentration parameter and (x·μ) is the dot product.

        Args:
            embedding (EmbeddingTensor): Document embedding vector (normalized).
            cluster_id (int): The cluster ID to calculate likelihood for.
                If the cluster doesn't exist and global_mean is None, returns 0.0.
                If the cluster doesn't exist but global_mean is available, uses
                global_mean.

        Returns:
            float: Log-likelihood of the document under the cluster's vMF distribution.
        """
        embedding = self._normalize(embedding)

        if cluster_id not in self.cluster_params:
            if self.global_mean is None:
                return 0.0
            return self.kappa * np.dot(embedding, self.global_mean)

        assert "mean" in self.cluster_params[cluster_id]
        cluster_mean = self.cluster_params[cluster_id]["mean"]

        return self.kappa * np.dot(embedding, cluster_mean)

    def log_crp_prior(self, cluster_id: Optional[int] = None) -> float:
        """
        Calculate the Chinese Restaurant Process prior probability.

        The Chinese Restaurant Process (CRP) is a stochastic process that defines
        a probability distribution over partitions of items. In the context of
        clustering, it provides a prior probability for assigning a document to
        an existing cluster or creating a new one.

        Args:
            cluster_id (Optional[int]): The cluster ID.
                If provided, calculate prior for an existing cluster.
                If None, calculate prior for a new cluster.

        Returns:
            float: Log probability of the cluster under the CRP prior.
        """
        total_documents = len(self.clusters)
        denominator = self.alpha + total_documents

        if cluster_id is None:
            return np.log(self.alpha / denominator)

        assert "mean" in self.cluster_params[cluster_id]
        count = self.cluster_params[cluster_id]["count"]
        return np.log(count / denominator)

    def log_likelihood(
        self, embedding: EmbeddingTensor
    ) -> tuple[dict[int, float], float]:
        """
        Calculate log-likelihoods of an embedding for all existing clusters and a new cluster.

        This method computes the log-likelihood of a document embedding under each
        existing cluster's von Mises-Fisher distribution, as well as under a potential
        new cluster.

        Args:
            embedding (EmbeddingTensor): Document embedding vector.

        Returns:
            tuple: A tuple containing:
                - dict[int, float]: Dictionary mapping cluster IDs to their log-likelihoods.
                - float: Log-likelihood for a new cluster.
        """  # noqa: E501
        embedding = self._normalize(embedding)
        existing_likelihoods = {}

        # Calculate likelihood for each existing cluster
        for cid in self.cluster_params:
            existing_likelihoods[cid] = self._log_likelihood_base(embedding, cid)

        # Calculate likelihood for a new cluster
        new_cluster_likelihood = self._log_likelihood_base(embedding, -1)

        return existing_likelihoods, new_cluster_likelihood

    def _calculate_cluster_probabilities(
        self, embedding: EmbeddingTensor
    ) -> tuple[list[Union[int, None]], np.ndarray]:
        """
        Calculate the probability distribution over clusters for a document.

        This method combines the CRP prior and von Mises-Fisher likelihood to get
        the posterior probability of cluster assignment according to the Chinese
        Restaurant Process. It computes probabilities for assigning the document
        to each existing cluster or creating a new one.

        Args:
            embedding (EmbeddingTensor): Document embedding vector.

        Returns:
            tuple: A tuple containing:
                - list[Union[int, None]]: List of existing cluster IDs, with None
                  representing a potential new cluster.
                - np.ndarray: Probability distribution over clusters (including new cluster).
        """  # noqa: E501
        embedding = self._normalize(embedding)

        cluster_ids = list(self.cluster_params.keys())
        existing_likelihoods, new_cluster_likelihood = self.log_likelihood(embedding)

        # Combine prior and likelihood for each cluster
        scores = []

        # Existing clusters
        for cid in cluster_ids:
            prior = self.log_crp_prior(cid)
            likelihood = existing_likelihoods[cid]
            scores.append(prior + likelihood)

        # New cluster
        prior_new = self.log_crp_prior()
        scores.append(prior_new + new_cluster_likelihood)

        # Convert log scores to probabilities
        scores = np.array(scores)
        scores -= logsumexp(scores)
        probabilities = np.exp(scores)  # type: np.ndarray

        # Add placeholder for new cluster ID
        extended_cluster_ids = cluster_ids + [None]  # None represents new cluster

        return extended_cluster_ids, probabilities

    def _create_or_update_cluster(
        self,
        embedding: EmbeddingTensor,
        is_new_cluster: bool,
        existing_cluster_id: Optional[int] = None,
    ) -> int:
        """
        Create a new cluster or update an existing one with a document.

        This method either creates a new cluster with the given embedding as its
        initial mean, or updates an existing cluster's parameters by incorporating
        the new embedding.

        Args:
            embedding (EmbeddingTensor): Document embedding vector.
            is_new_cluster (bool): Whether to create a new cluster.
            existing_cluster_id (Optional[int]): ID of existing cluster to update,
                if is_new_cluster is False.

        Returns:
            int: The ID of the created or updated cluster.
        """
        # TODO: Looks like we can just check if existing_cluster_id is None
        #       instead of having is_new_cluster flag
        if is_new_cluster:
            # Create new cluster
            cid = self.next_id
            # Convert to numpy array to ensure compatibility
            embedding_np = to_numpy(embedding)
            self.cluster_params[cid] = {"mean": embedding_np, "count": 1}
            self.next_id += 1
            self.clusters.append(cid)
            return cid

        # Update existing cluster
        assert existing_cluster_id is not None
        cid = existing_cluster_id
        params = self.cluster_params[cid]
        params["count"] += 1
        params["mean"] = self._normalize(
            params["mean"] * (params["count"] - 1) + embedding
        )
        self.clusters.append(cid)

        return cid

    def assign_cluster(self, embedding: EmbeddingTensor) -> tuple[int, np.ndarray]:
        """
        Assign a document embedding to a cluster using Bayesian inference.

        This method computes probabilities for assigning the document to each
        existing cluster or creating a new one, then samples a cluster assignment
        from this probability distribution. The probabilities combine the CRP prior
        and the von Mises-Fisher likelihood.

        Args:
            embedding (EmbeddingTensor): Document embedding vector.

        Returns:
            tuple: A tuple containing:
                - int: The assigned cluster ID.
                - np.ndarray: Probability distribution over clusters used for assignment.
        """  # noqa: E501
        # Calculate probabilities over all clusters (including a possible new one)
        extended_cluster_ids, probs = self._calculate_cluster_probabilities(embedding)

        # Sample a cluster according to the probabilities
        chosen = self.random_state.choice(len(probs), p=probs)

        # Check if we need to create a new cluster
        is_new_cluster = (
            chosen == len(extended_cluster_ids) - 1
        )  # Last index represents new cluster

        if is_new_cluster:
            # Create new cluster
            cluster_id = self._create_or_update_cluster(embedding, is_new_cluster=True)
        else:
            # Update existing cluster
            cluster_id = self._create_or_update_cluster(
                embedding,
                is_new_cluster=False,
                existing_cluster_id=extended_cluster_ids[chosen],
            )

        return cluster_id, probs

    def fit(self, documents, y: Union[Any, None] = None):
        """
        Train the clustering model on the given text data.

        This method processes each document in the input, computing its embedding
        and assigning it to a cluster using Bayesian inference with the Chinese
        Restaurant Process. It supports both text inputs and pre-computed embeddings.

        Args:
            documents: array-like of shape (n_samples,)
                The text documents or embeddings to cluster.
            y: Ignored. Added for compatibility with scikit-learn API.

        Returns:
            self: The fitted model instance.

        Side effects:
            Sets self.embeddings_ with the document embeddings.
            Sets self.labels_ with the cluster assignments.
            Updates self.clusters and self.cluster_params with cluster information.
        """
        # Generate embeddings from text
        if isinstance(documents[0], str):
            self.embeddings_ = self.get_embedding(documents)
        else:
            # If embeddings are provided directly
            self.embeddings_ = np.array([self._normalize(e) for e in documents])

        # Calculate global mean from normalized embeddings
        self.global_mean = np.mean(self.embeddings_, axis=0)

        # Reset clustering state
        self.clusters = []
        self.cluster_params = {}
        self.next_id = 0

        # Assign all documents to clusters
        assignments = []
        for emb in self.embeddings_:
            cid, _ = self.assign_cluster(emb)
            assignments.append(cid)

        self.labels_ = np.array(assignments)
        return self

    def predict(self, documents):
        """
        Predict the closest cluster for each sample in documents.

        This method computes the most likely cluster assignment for each document
        based on the von Mises-Fisher likelihood, without updating the cluster
        parameters. It supports both text inputs and pre-computed embeddings.

        Args:
            documents: array-like of shape (n_samples,)
                The text documents or embeddings to predict clusters for.

        Returns:
            ndarray of shape (n_samples,): Cluster labels for each document.
                Returns -1 if no clusters exist yet.
        """
        # Generate embeddings from text
        if isinstance(documents[0], str):
            embeddings = self.get_embedding(documents)
        else:
            # If embeddings are provided directly
            embeddings = np.array([self._normalize(e) for e in documents])

        predictions = []
        for emb in embeddings:
            # For prediction, we use a deterministic approach (max probability)
            scores = []
            for cid in self.cluster_params:
                likelihood = self._log_likelihood_base(emb, cid)
                scores.append((cid, likelihood))

            if not scores:  # If no clusters exist yet
                predictions.append(-1)
            else:
                best_cluster = max(scores, key=lambda x: x[1])[0]
                predictions.append(best_cluster)

        return np.array(predictions)

    def fit_predict(self, documents, y: Union[Any, None] = None):
        """
        Fit the model and predict cluster labels for documents.

        This method is a convenience function that calls fit() followed by
        returning the cluster labels from the fitting process.

        Args:
            documents: array-like of shape (n_samples,)
                The text documents or embeddings to cluster.
            y: Ignored. Added for compatibility with scikit-learn API.

        Returns:
            ndarray of shape (n_samples,): Cluster labels for each document.
        """
        self.fit(documents)
        return self.labels_


class DirichletProcessOld:
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
        sigma: float = 0.0,
        base_measure: Optional[dict] = None,
        similarity_metric: Optional[
            Callable[[EmbeddingTensor, EmbeddingTensor], float]
        ] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize a Dirichlet Process clustering model.

        Args:
            alpha (float): Concentration parameter for new cluster creation.
                Higher values lead to more clusters.
            sigma (float): Discount parameter. Set to 0.0 and not used for Dirichlet
                Process, but declared to allow proper inheritance by Pitman-Yor
                Process.
            base_measure (Optional[dict]): Base measure parameters for the DP.
                Should contain 'variance' key for the likelihood model.
            similarity_metric (Optional[Callable]): Function to compute similarity
                between embeddings. If None, uses cosine_similarity.
            random_state (Optional[int]): Random seed for reproducibility.
                If None, then fresh, unpredictable entropy will be pulled from the OS.

        Raises:
            TypeError: If base_measure is not a dict or does not contain 'variance' key.
        """
        self.alpha = alpha
        _ = sigma  # Help linters understand that sigma is not used in this class

        self.clusters: list[int] = []
        self.cluster_params: dict[int, dict] = {}
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.similarity_metric = (
            similarity_metric if similarity_metric else self.cosine_similarity
        )
        self.embedding_dim = None  # Will be set on first embedding

        # For reproducibility
        self.random_state = np.random.default_rng(seed=random_state)

        # For tracking processed texts and their embeddings
        self.text_embeddings: dict[str, EmbeddingTensor] = {}

        # Ensure base_measure has a variance value
        if base_measure is None or "variance" not in base_measure:
            # We expect variance from the user, and this is a moderate variance
            # default value. It's not expected to use this default value in practice.
            self.base_measure: dict[Literal["variance"], float] = {"variance": 0.3}
        elif not isinstance(base_measure["variance"], float):
            # Ensure the variance is a float
            raise TypeError(
                "The variance in base_measure must be a float, got "
                f"{type(base_measure['variance'])}"
            )
        else:
            self.base_measure: dict[Literal["variance"], float] = base_measure

    def get_embedding(self, text: str) -> EmbeddingTensor:
        """
        Get the embedding for a text.

        Args:
            text (str): The text to embed.

        Returns: The embedding vector for the text.
        """
        # Check if already computed in this session
        if text in self.text_embeddings:
            return self.text_embeddings[text]

        embedding = self.model.encode(text, show_progress_bar=False)

        # Set embedding dimension if not set
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)

        # Store in session cache
        self.text_embeddings[text] = embedding

        return embedding

    def cosine_similarity(
        self, embedding1: EmbeddingTensor, embedding2: EmbeddingTensor
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding.
            embedding2: Second embedding.

        Returns:
            float: Similarity score between 0 and 1, where 1 means identical.
        """
        similarity = 1 - cosine(embedding1, embedding2)
        return max(0.0, similarity)

    def log_likelihood(self, embedding: EmbeddingTensor, cluster_id: int) -> float:
        """
        Calculate log likelihood of an embedding under a cluster's distribution.

        This implements a multivariate Gaussian likelihood in the embedding space.

        Args:
            embedding: The embedding to evaluate.
            cluster_id: The cluster ID.

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

    def _log_likelihood_base_measure(self, embedding: EmbeddingTensor) -> float:
        """
        Calculate log likelihood of an embedding under the base measure.

        Args:
            embedding: The embedding to evaluate.

        Returns:
            float: Log likelihood of the embedding under the base measure.
        """
        # For the base measure, we use a wider variance
        # Ensure we have a valid variance value
        variance = self.base_measure["variance"] * 10.0

        # For new clusters, we center at the embedding itself
        if self.embedding_dim is not None:
            dim = float(self.embedding_dim)
        else:
            # Fallback if embedding_dim is not set
            dim = float(len(embedding))

        # The same value as in the likelihood function ???
        return -0.5 * dim * np.log(2 * np.pi * variance)

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

        # Otherwise its new cluster
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
            self.cluster_params[new_cluster_id] = {
                "mean": to_numpy(embedding),
                "count": 1,
            }

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
        variance = self.base_measure["variance"]

        # Test: 1
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
            self.cluster_params[chosen_cluster_id] = {
                "mean": to_numpy(embedding),
                "count": 1,
            }
        else:
            # Update existing cluster
            current_mean = self.cluster_params[chosen_cluster_id]["mean"]
            current_count = self.cluster_params[chosen_cluster_id]["count"]
            emb_array = to_numpy(embedding)

            # Update mean using online update formula
            new_mean = (current_mean * current_count + emb_array) / (current_count + 1)

            self.cluster_params[chosen_cluster_id]["mean"] = new_mean
            self.cluster_params[chosen_cluster_id]["count"] += 1

        return chosen_cluster_id

    def fit(self, texts: list[str]) -> tuple[list[int], dict]:
        """
        Train the clustering model on the given text data.

        This method processes each text in the input list, assigning it to a cluster
        using Bayesian inference. For DirichletProcess, it uses the Chinese Restaurant
        Process prior. For PitmanYorProcess, it uses the Pitman-Yor Process prior.

        The method automatically detects which model is being used based on the class
        and applies the appropriate clustering algorithm.

        Args:
            texts (list[str]): List of text strings to cluster.

        Returns:
            tuple[list[int], dict]: A tuple containing:
                - List of cluster assignments for each text
                - Dictionary of cluster parameters
        """
        logger.info("Start processing %d texts ...", len(texts))

        # Reset state for a fresh run
        self.clusters = []
        self.cluster_params = {}

        def format_process(class_name: str) -> str:
            """
            Format a class name into a human-readable progress description.

            Converts CamelCase class names to space-separated words and handles
            special cases like 'PitmanYorProcess' to 'Pitman-Yor Process'.

            Args:
                class_name (str): The name of the class to format

            Returns:
                str: Formatted string with timestamp and readable class name
                     in the format:
                     "YYYY-MM-DD HH:MM:SS - INFO - Clustering with {formatted_name}"
            """
            import re
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_name = (
                "Pitman-Yor Process"  # Special case for Pitman-Yor
                if class_name == "PitmanYorProcess"
                else re.sub(r"(?<!^)(?=[A-Z])", " ", class_name)
            )

            return f"{timestamp} - INFO - Clustering with {formatted_name}"

        # Process all texts with a single progress bar
        for text in tqdm(
            texts,
            desc=format_process(self.__class__.__name__),
            total=len(texts),
            disable=None,  # Disable on non-TTY
            unit=" texts",
        ):
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
        clusters (list[int]): List of cluster assignments for each processed text.
        cluster_params (dict): Dictionary of cluster parameters for each cluster.
        model: Sentence transformer model used for text embeddings.
    """

    def __init__(
        self,
        alpha: float,
        sigma: float,
        base_measure: Optional[dict] = None,
        similarity_metric: Optional[
            Callable[[EmbeddingTensor, EmbeddingTensor], float]
        ] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize a Pitman-Yor Process clustering model.

        Args:
            alpha (float): Concentration parameter controlling the propensity to
                create new clusters. Higher values lead to more clusters. Must
                satisfy: alpha > -sigma.
            sigma (float): Discount parameter controlling power-law behavior.
                The value must satisfy: sigma ∈ [0.0, 1.0). As sigma approaches
                1.0, the distribution exhibits heavier tails; sigma = 0.0 corresponds
                to the :class:`DirichletProcess`, which is irrelevant for this
                implementation.
            base_measure (Optional[dict]): Base measure parameters for the PYP.
                Should contain 'variance' key for the likelihood model.
            similarity_metric (Optional[Callable]): Function to compute similarity
                between embeddings. If None, uses
                :meth:`DirichletProcess.cosine_similarity`.
            random_state (Optional[int]): Random seed for reproducibility.

        Raises:
            ValueError: If sigma ∉ [0.0, 1.0) or if alpha ≤ -sigma.
            TypeError: If base_measure is not a dict or does not contain 'variance' key.
        """
        if sigma < 0.0 or sigma >= 1.0:
            raise ValueError(
                f"Discount parameter sigma must be in the interval [0.0, 1.0); "
                f"got {sigma}"
            )

        if alpha <= -sigma:
            raise ValueError(
                f"Parameter alpha must be greater than -sigma (i.e., alpha > {-sigma}) "
                f"for sigma={sigma}"
            )

        super().__init__(alpha, 0.0, base_measure, similarity_metric, random_state)
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

        # Otherwise it's new cluster
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
            self.cluster_params[new_cluster_id] = {
                "mean": to_numpy(embedding),
                "count": 1,
            }

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
        variance = self.base_measure["variance"]
        # The same value as in the likelihood function ???
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
            self.cluster_params[chosen_cluster_id] = {
                "mean": to_numpy(embedding),
                "count": 1,
            }
        else:
            # Update existing cluster
            current_mean = self.cluster_params[chosen_cluster_id]["mean"]
            current_count = self.cluster_params[chosen_cluster_id]["count"]
            emb_array = to_numpy(embedding)

            # Update mean using online update formula
            new_mean = (current_mean * current_count + emb_array) / (current_count + 1)

            self.cluster_params[chosen_cluster_id]["mean"] = new_mean
            self.cluster_params[chosen_cluster_id]["count"] += 1

        return chosen_cluster_id
