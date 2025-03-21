"""
Clustering models for text data using Dirichlet Process and Pitman-Yor Process.

This module implements nonparametric Bayesian clustering algorithms for text data,
specifically the Dirichlet Process and Pitman-Yor Process. These methods automatically
determine the appropriate number of clusters based on the data.

The implementation uses the Chinese Restaurant Process formulation with
von Mises-Fisher distribution for modeling document embeddings on the unit hypersphere.

Classes
-------
:class:`DirichletProcess`
    Implements clustering using the Dirichlet Process with concentration parameter
    alpha and precision parameter kappa.
:class:`PitmanYorProcess`
    Extends DirichletProcess with an additional discount parameter for more flexible
    power-law behavior in cluster size distributions.

Notes
-----
Both implementations follow a scikit-learn compatible API with ``fit()``,
``predict()``, and ``fit_predict()`` methods. The Pitman-Yor Process is generally
better suited for text data as it can model the power-law distributions common in
natural language.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from typing import Optional, Union

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# TODO: Get rid of assert statements, use type checking and raise errors
class DirichletProcess:
    """
    DP clustering implementation for text data using von Mises-Fisher distribution.

    This implementation uses a Chinese Restaurant Process (CRP) formulation with
    Bayesian inference to cluster text data. It combines the CRP prior with
    a likelihood model based on von Mises-Fisher distributions in the embedding
    space, which is particularly suitable for directional data like normalized
    text embeddings.

    The model uses a concentration parameter alpha to control the propensity to
    create new clusters, and a precision parameter kappa to control the
    concentration of points around cluster means in the von Mises-Fisher distribution.

    Attributes
    ----------
    alpha : float
        Concentration parameter for new cluster creation.
    kappa : float
        Precision parameter for the von Mises-Fisher distribution.
    model : SentenceTransformer
        Sentence transformer model used for text embeddings.
    random_state : numpy.random.Generator
        Random state for reproducibility.
    clusters : list of int
        List of cluster assignments for each processed text.
    cluster_params : dict[int, dict[str, NDArray[np.float32] | int]]
        Dictionary of cluster parameters for each cluster.
        Contains 'mean' (centroid) and 'count' (number of points).
    global_mean : Optional[NDArray[np.float32]]
        Global mean of all document embeddings.
    next_id : int
        Next available cluster ID.
    embeddings_ : Optional[NDArray[np.float32]]
        Document embeddings after fitting.
    labels_ : Optional[NDArray[np.int64]]
        Cluster assignments after fitting.
    text_embeddings : dict[str, NDArray[np.float32]]
        Cache of text to embedding mappings.
    embedding_dim : Optional[int]
        Dimension of the embedding vectors.
    """

    def __init__(
        self,
        alpha: float,
        kappa: float,
        model_name: Optional[str] = "all-MiniLM-L6-v2",
        random_state: Optional[int] = None,
    ):
        """
        Initialize a Dirichlet Process model with von Mises-Fisher likelihood.

        Parameters
        ----------
        alpha : float
            Concentration parameter for new cluster creation.
            Higher values lead to more clusters.
        kappa : float
            Precision parameter for the von Mises-Fisher distribution.
            Higher values lead to tighter, more concentrated clusters.
        model_name : Optional[str]
            Name of the sentence transformer model to use.
            Default is "all-MiniLM-L6-v2".
        random_state : Optional[int]
            Random seed for reproducibility.
            If None, fresh, unpredictable entropy will be pulled from the OS.
        """
        self.alpha = alpha
        self.kappa = kappa
        self.model = SentenceTransformer(model_name or "all-MiniLM-L6-v2")

        # For reproducibility
        self.random_state = np.random.default_rng(seed=random_state)

        self.clusters: list[int] = []
        self.cluster_params: dict[int, dict[str, NDArray[np.float32] | int]] = {}
        self.global_mean: Optional[NDArray[np.float32]] = None
        self.next_id = 0
        self.embeddings_: Optional[NDArray[np.float32]] = None
        self.labels_: Optional[NDArray[np.int64]] = None

        # For tracking processed texts and their embeddings
        self.text_embeddings: dict[str, NDArray[np.float32]] = {}
        self.embedding_dim: Optional[int] = None  # Will be set on first embedding

    def get_embedding(self, text: Union[str, list[str]]) -> NDArray[np.float32]:
        """
        Get the embedding for a text or list of texts with caching.

        This method computes embeddings for text inputs using the sentence transformer
        model. It implements caching to avoid recomputing embeddings for previously
        seen texts. The method can handle both single text strings and lists of texts.

        Parameters
        ----------
        text : str or list of str
            Text or list of texts to embed.

        Returns
        -------
        NDArray[np.float32]
            The normalized embedding vector(s) for the text.
            If input is a single string, returns a single embedding vector.
            If input is a list, returns an array of embedding vectors.
        """
        # Handle single text vs list
        is_single = isinstance(text, str)
        if isinstance(text, str):
            texts: list[str] = [text]
        else:
            texts = text

        # Get embeddings (using cache when available)
        results: list[NDArray[np.float32]] = []
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

    def _normalize(self, embedding: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Normalize vector to unit length for use with von Mises-Fisher distribution.

        The von Mises-Fisher distribution is defined on the unit hypersphere, so
        all vectors must be normalized to unit length.

        Parameters
        ----------
        embedding : NDArray[np.float32]
            The embedding vector to normalize.

        Returns
        -------
        NDArray[np.float32]
            The normalized embedding vector with unit length as a NumPy array.
        """
        norm = np.linalg.norm(embedding)
        # Convert to numpy array to ensure division works properly
        embedding_np = np.asarray(embedding)
        # Ensure the result is float32 to match the return type
        result = embedding_np / norm if norm > 0 else embedding_np
        return result.astype(np.float32)

    def _log_likelihood_vmf(
        self, embedding: NDArray[np.float32], cluster_id: int
    ) -> float:
        """
        Calculate von Mises-Fisher log-likelihood for a document in a cluster.

        The von Mises-Fisher distribution is a probability distribution on the
        unit hypersphere. For unit vectors x and μ, the log-likelihood is proportional
        to κ·(x·μ), where κ is the concentration parameter and (x·μ) is the dot product.

        Parameters
        ----------
        embedding : NDArray[np.float32]
            Document embedding vector (normalized).
        cluster_id : int
            The cluster ID to calculate likelihood for.
            If the cluster doesn't exist and global_mean is None, returns 0.0.
            If the cluster doesn't exist but global_mean is available, uses
            global_mean.

        Returns
        -------
        float
            Log-likelihood of the document under the cluster's vMF distribution.
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

        The CRP is a key component of Bayesian nonparametric models, particularly
        the Dirichlet Process. For more details, see [1]_, [2]_.

        Parameters
        ----------
        cluster_id : Optional[int]
            The cluster ID.
            If provided, calculate prior for an existing cluster.
            If None, calculate prior for a new cluster.

        Returns
        -------
        float
            Log probability of the cluster under the CRP prior.

        References
        ----------
        .. [1] Aldous, D. J. (1985). "Exchangeability and related topics."
               In École d'Été de Probabilités de Saint-Flour XIII—1983, pp. 1-198.
               Springer, Berlin, Heidelberg.
               https://doi.org/10.1007/BFb0099421

        .. [2] Teh, Y. W. (2010). "Dirichlet Process."
               In Encyclopedia of Machine Learning, pp. 280-287.
               Springer, Boston, MA.
               https://doi.org/10.1007/978-0-387-30164-8_219
        """
        total_documents = len(self.clusters)
        denominator = self.alpha + total_documents

        if cluster_id is None:
            return np.log(self.alpha / denominator)

        assert "mean" in self.cluster_params[cluster_id]
        count = self.cluster_params[cluster_id]["count"]
        return float(np.log(count / denominator))

    def log_likelihood(
        self, embedding: NDArray[np.float32]
    ) -> tuple[dict[int, float], float]:
        """
        Calculate log-likelihoods for an embedding across all clusters.

        This method computes the log-likelihood of a document embedding under each
        existing cluster's von Mises-Fisher distribution, as well as under a potential
        new cluster.

        Parameters
        ----------
        embedding : NDArray[np.float32]
            Document embedding vector.

        Returns
        -------
        existing_likelihoods : dict[int, float]
            Dictionary mapping cluster IDs to their log-likelihoods.
        new_cluster_likelihood : float
            Log-likelihood for a new cluster.
        """
        embedding = self._normalize(embedding)
        existing_likelihoods = {}

        # Calculate likelihood for each existing cluster
        for cid in self.cluster_params:
            existing_likelihoods[cid] = self._log_likelihood_vmf(embedding, cid)

        # Calculate likelihood for a new cluster
        new_cluster_likelihood = self._log_likelihood_vmf(embedding, -1)

        return existing_likelihoods, new_cluster_likelihood

    def _calculate_cluster_probabilities(
        self, embedding: NDArray[np.float32]
    ) -> tuple[list[Union[int, None]], np.ndarray]:
        """
        Calculate the probability distribution over clusters for a document.

        This method combines the CRP prior and von Mises-Fisher likelihood to get
        the posterior probability of cluster assignment according to the Chinese
        Restaurant Process. It computes probabilities for assigning the document
        to each existing cluster or creating a new one.

        Parameters
        ----------
        embedding : NDArray[np.float32]
            Document embedding vector.

        Returns
        -------
        cluster_ids : list[Union[int, None]]
            List of existing cluster IDs, with None representing a potential new
            cluster.
        probabilities : np.ndarray
            Probability distribution over clusters (including new cluster).
        """
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
        scores_array = np.array(scores)
        scores_array -= logsumexp(scores_array)  # type: ignore
        probabilities = np.exp(scores_array)

        # Add placeholder for new cluster ID
        extended_cluster_ids = cluster_ids + [None]  # None represents new cluster

        return extended_cluster_ids, probabilities

    def _create_or_update_cluster(
        self,
        embedding: NDArray[np.float32],
        is_new_cluster: bool,
        existing_cluster_id: Optional[int] = None,
    ) -> int:
        """
        Create a new cluster or update an existing one with a document.

        This method either creates a new cluster with the given embedding as its
        initial mean, or updates an existing cluster's parameters by incorporating
        the new embedding.

        Parameters
        ----------
        embedding : NDArray[np.float32]
            Document embedding vector.
        is_new_cluster : bool
            Whether to create a new cluster.
        existing_cluster_id : Optional[int]
            ID of existing cluster to update, if is_new_cluster is False.

        Returns
        -------
        int
            The ID of the created or updated cluster.
        """
        # TODO: Looks like we can just check if existing_cluster_id is None
        #       instead of having is_new_cluster flag
        if is_new_cluster:
            # Create new cluster
            cid = self.next_id
            # Convert to numpy array to ensure compatibility
            embedding_np = np.asarray(embedding)
            self.cluster_params[cid] = {"mean": embedding_np, "count": 1}
            self.next_id += 1
            self.clusters.append(cid)
            return cid

        # Update existing cluster
        assert existing_cluster_id is not None
        params = self.cluster_params[existing_cluster_id]
        params["count"] += 1

        result = (params["mean"] * (params["count"] - 1) + embedding).astype(np.float32)
        params["mean"] = self._normalize(result)
        self.clusters.append(existing_cluster_id)

        return existing_cluster_id

    def assign_cluster(self, embedding: NDArray[np.float32]) -> tuple[int, np.ndarray]:
        """
        Assign a document embedding to a cluster using Bayesian inference.

        This method computes probabilities for assigning the document to each
        existing cluster or creating a new one, then samples a cluster assignment
        from this probability distribution. The probabilities combine the CRP prior
        and the von Mises-Fisher likelihood.

        Parameters
        ----------
        embedding : NDArray[np.float32]
            Document embedding vector.

        Returns
        -------
        cluster_id : int
            The assigned cluster ID.
        probs : np.ndarray
            Probability distribution over clusters used for assignment.
        """
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

    def fit(self, documents, _y=None):
        """
        Train the clustering model on the given text data.

        This method processes each document in the input, computing its embedding
        and assigning it to a cluster using Bayesian inference with the Chinese
        Restaurant Process. It supports both text inputs and pre-computed embeddings.

        Parameters
        ----------
        documents : Union[list[str], list[NDArray[np.float32]]]
            The text documents or embeddings to cluster.
        _y
            Ignored. Added for compatibility with scikit-learn API.

        Returns
        -------
        self : object
            The fitted estimator.

        Note
        ----
        After fitting, the following attributes are set:

        - :data:`embeddings_` : Optional[NDArray[np.float32]]
            The document embeddings.
        - :data:`labels_` : NDArray[np.int64]
            The cluster assignments for each document.
        - :data:`clusters` : list
            List of cluster IDs for each document.
        - :data:`cluster_params` : dict
            Dictionary containing parameters for each cluster.
        """
        # TODO: from tqdm.auto import tqdm
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

        Parameters
        ----------
        documents : Union[list[str], list[NDArray[np.float32]]]
            The text documents or embeddings to predict clusters for.

        Returns
        -------
        labels : NDArray[np.int64]
            Cluster labels for each document.
            Returns -1 if no clusters exist yet.

        Note
        ----
        This method computes the most likely cluster assignment for each document
        based on the von Mises-Fisher likelihood, without updating the cluster
        parameters. It supports both text inputs and pre-computed embeddings.
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
                likelihood = self._log_likelihood_vmf(emb, cid)
                scores.append((cid, likelihood))

            if not scores:  # If no clusters exist yet
                predictions.append(-1)
            else:
                best_cluster = max(scores, key=lambda x: x[1])[0]
                predictions.append(best_cluster)

        return np.array(predictions)

    def fit_predict(self, documents, _y=None):
        """
        Fit the model and predict cluster labels for documents.

        Parameters
        ----------
        documents : Union[list[str], list[NDArray[np.float32]]]
            The text documents or embeddings to cluster.
        _y
            This parameter exists only for compatibility with scikit-learn API.

        Returns
        -------
        labels : NDArray[np.int64]
            Cluster labels for each document.

        Notes
        -----
        This method is a convenience function that calls :func:`fit` followed by
        returning the cluster labels from the fitting process.
        """
        self.fit(documents)
        return self.labels_


class PitmanYorProcess(DirichletProcess):
    """
    Pitman-Yor Process clustering for text data using von Mises-Fisher distribution.

    Attributes
    ----------
    alpha : float
        Concentration parameter for new cluster creation.
    kappa : float
        Precision parameter for the von Mises-Fisher distribution.
    sigma : float
        Discount parameter controlling power-law behavior (0 ≤ σ < 1).
    model : SentenceTransformer
        Sentence transformer model used for text embeddings.
    random_state : numpy.random.Generator
        Random state for reproducibility.
    clusters : list[int]
        List of cluster assignments for each processed text.
    cluster_params : dict[int, dict[str, NDArray[np.float32] | int]]
        Dictionary of cluster parameters for each cluster.
        Contains 'mean' (centroid) and 'count' (number of points).
    global_mean : Optional[NDArray[np.float32]]
        Global mean of all document embeddings.
    next_id : int
        Next available cluster ID.
    embeddings_ : Optional[NDArray[np.float32]]
        Document embeddings after fitting.
    labels_ : Optional[NDArray[np.int64]]
        Cluster assignments after fitting.
    text_embeddings : dict[str, NDArray[np.float32]]
        Cache of text to embedding mappings.
    embedding_dim : Optional[int]
        Dimension of the embedding vectors.

    Notes
    -----
    The Pitman-Yor Process is a generalization of the Dirichlet Process that introduces
    a discount parameter (sigma) to control the power-law behavior of the cluster
    size distribution. It is particularly effective for modeling natural language
    phenomena that exhibit power-law distributions, such as word frequencies or
    topic distributions.

    This implementation extends the DirichletProcess class, adding the sigma parameter
    and modifying the cluster assignment probabilities according to the Pitman-Yor
    Process while maintaining the von Mises-Fisher likelihood model for directional
    text embeddings.

    The mathematical foundation of the Pitman-Yor Process involves two key parameters:

    - The concentration parameter alpha (α > -σ), controlling the overall
      tendency to create new clusters
    - The discount parameter sigma (0 ≤ σ < 1), controlling the power-law behavior

    As σ approaches 1, the distribution exhibits heavier tails (more small clusters),
    while σ = 0 reduces to the standard Dirichlet Process.
    """

    def __init__(
        self,
        alpha: float,
        kappa: float,
        sigma: float,
        model_name: Optional[str] = "all-MiniLM-L6-v2",
        random_state: Optional[int] = None,
    ):
        """
        Initialize a PYP clustering model with von Mises-Fisher likelihood.

        Parameters
        ----------
        alpha : float
            Concentration parameter for the Pitman-Yor Process.
            Higher values encourage formation of more clusters.
            Must satisfy: α > -σ.
        kappa : float
            Precision parameter for the von Mises-Fisher distribution.
            Higher values lead to tighter, more concentrated clusters.
        sigma : float
            Discount parameter for the Pitman-Yor Process (0 ≤ σ < 1).
            Controls the power-law behavior. Higher values create more
            power-law-like cluster size distributions. When σ=0, the model
            reduces to a Dirichlet Process.
        model_name : Optional[str]
            Name of the sentence transformer model to use.
            Default is "all-MiniLM-L6-v2".
        random_state : Optional[int]
            Random seed for reproducibility.
            If None, fresh, unpredictable entropy will be pulled from the OS.

        Raises
        ------
        ValueError
            If sigma ∉ [0.0, 1.0) or if alpha ≤ -sigma.

        Notes
        -----
        The mathematical requirement for the Pitman-Yor Process is:

        - The discount parameter σ must be in [0,1)
        - The concentration parameter α must satisfy α > -σ

        The constraint α > -σ ensures that the numerator in the new
        table probability calculation (α + K*σ) remains positive even
        when K=0. This is essential for proper probabilistic behavior of the model.
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

        super().__init__(
            alpha=alpha, kappa=kappa, model_name=model_name, random_state=random_state
        )
        self.sigma = sigma

    def log_pyp_prior(self, cluster_id: Optional[int] = None) -> float:
        """
        Calculate the Pitman-Yor Process prior probability.

        Parameters
        ----------
        cluster_id : int or None
            The cluster ID.
            If provided, calculate prior for an existing cluster.
            If None, calculate prior for a new cluster.

        Returns
        -------
        float
            Log probability of the cluster under the PYP prior.

        Notes
        -----
        The Pitman-Yor Process generalizes the Chinese Restaurant Process with the
        introduction of a discount parameter σ. The probability of a new customer
        (document) joining an existing table (cluster) k or starting a new table is:

        P(existing cluster k) = (n_k - σ) / (n + α)
        P(new cluster) = (α + K*σ) / (n + α)

        where:
        - n_k is the number of customers at table k
        - n is the total number of customers
        - K is the current number of tables
        - σ is the discount parameter
        - α is the concentration parameter
        """
        total_documents = len(self.clusters)

        # If no documents processed yet, return uniform prior
        if total_documents == 0:
            return 0.0

        # Number of existing clusters/tables
        num_clusters = len(self.cluster_params)
        denominator = total_documents + self.alpha

        if cluster_id is None:
            # Prior for a new cluster: (alpha + K*sigma) / (n + alpha)
            numerator = self.alpha + num_clusters * self.sigma
            return np.log(numerator / denominator)

        # Prior for an existing cluster: (n_k - sigma) / (n + alpha)
        assert "count" in self.cluster_params[cluster_id]
        count = int(self.cluster_params[cluster_id]["count"])
        numerator = count - self.sigma

        # If numerator is negative or zero, use a small positive value
        if numerator <= 0:
            numerator = 1e-10

        return float(np.log(numerator / denominator))

    def _calculate_cluster_probabilities(
        self, embedding: NDArray[np.float32]
    ) -> tuple[list[Union[int, None]], np.ndarray]:
        """
        Calculate the probability distribution over clusters for a document using PYP.

        This method combines the PYP prior and von Mises-Fisher likelihood to get
        the posterior probability of cluster assignment. It computes probabilities
        for assigning the document to each existing cluster or creating a new one.

        The key difference from the Dirichlet Process is the use of the Pitman-Yor
        prior, which introduces the discount parameter σ to create power-law behavior
        in the cluster size distribution.

        Parameters
        ----------
        embedding : NDArray[np.float32]
            Document embedding vector.

        Returns
        -------
        tuple
            A tuple containing:
            - list[Union[int, None]]: List of existing cluster IDs, with None
              representing a potential new cluster.
            - np.ndarray: Probability distribution over clusters
              (including new cluster).
        """
        # Normalize input vector
        embedding = self._normalize(embedding)

        # Get existing cluster IDs
        cluster_ids = list(self.cluster_params.keys())

        # Calculate likelihoods
        existing_likelihoods, new_cluster_likelihood = self.log_likelihood(embedding)

        # Combine prior and likelihood for each cluster
        scores = []

        # Existing clusters
        for cid in cluster_ids:
            prior = self.log_pyp_prior(cid)  # Use PYP prior instead of CRP
            likelihood = existing_likelihoods[cid]
            scores.append(prior + likelihood)

        # New cluster
        prior_new = self.log_pyp_prior()  # Use PYP prior for new cluster
        scores.append(prior_new + new_cluster_likelihood)

        # Convert log scores to probabilities
        scores_array = np.array(scores)
        scores_array -= logsumexp(scores_array)  # type: ignore
        probabilities = np.exp(scores_array)

        # Add placeholder for new cluster ID
        extended_cluster_ids = cluster_ids + [None]  # None represents new cluster

        return extended_cluster_ids, probabilities
