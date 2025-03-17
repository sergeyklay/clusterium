========================
Methodological Framework
========================

This document describes the methodological decisions and theoretical considerations underlying the Clusterium implementation. It explains how key algorithms are implemented, why specific  approaches were chosen, and their academic foundations.

Clustering algorithms
---------------------

This section documents the design and implementation of the nonparametric Bayesian clustering algorithms in Clusterium.

Dirichlet Process Clustering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clusterium implements text clustering using the Dirichlet Process (DP), a fundamental nonparametric Bayesian model that allows for a flexible, potentially infinite number of clusters. Unlike traditional clustering algorithms that require pre-specifying the number of clusters (e.g., K-means), the DP automatically determines the appropriate number of clusters based on the data. The theoretical foundations for this approach were established by Ferguson [1]_.

**Mathematical Foundation:**

In Clusterium's implementation, the DP is realized through the Chinese Restaurant Process (CRP) formulation. The prior probability of a document joining an existing cluster or creating a new one follows:

.. math::

   \log P(\text{document joins cluster } k) = \log\frac{n_k}{n + \alpha}

   \log P(\text{document forms new cluster}) = \log\frac{\alpha}{n + \alpha}

where ``n`` is the total number of documents and ``n_k`` is the number of documents in cluster ``k``.

Document representation in Clusterium is achieved through the SentenceTransformer library to generate text embeddings. These embeddings are normalized to unit length, facilitating the application of the von Mises-Fisher (vMF) distribution as the likelihood model:

.. math::

   \log p(x | \mu_k, \kappa) = \kappa \cdot (x \cdot \mu_k)

where the log-likelihood is computed as a dot product between the document embedding and the cluster mean, scaled by the concentration parameter.

**Theoretical Justification for von Mises-Fisher Distribution:**

The choice of von Mises-Fisher distribution for modeling document embeddings is grounded in several theoretical considerations as shown by Banerjee et al. [2]_. The full probability density function of the vMF distribution is:

.. math::

   f(x; \mu, \kappa) = C_d(\kappa) \exp(\kappa \mu^T x)

where :math:`C_d(\kappa)` is the normalization constant:

.. math::

   C_d(\kappa) = \frac{\kappa^{d/2-1}}{(2\pi)^{d/2}I_{d/2-1}(\kappa)}

and :math:`I_v` is the modified Bessel function of the first kind at order :math:`v`.

In Clusterium's implementation, we omit the normalization constant for computational efficiency since it does not affect the relative probabilities between clusters when using a fixed :math:`\kappa`. This decision is justified because:

1. Text embeddings naturally reside on the unit hypersphere after normalization, making directional statistics appropriate
2. Cosine similarity, which is the primary comparison metric in semantic text spaces, directly corresponds to the dot product of normalized vectors
3. The vMF distribution focuses on the direction of vectors rather than their magnitude, aligning with how semantic similarity functions in natural language processing
4. The concentration parameter :math:`\kappa` provides a theoretically sound way to control cluster tightness

These properties make vMF particularly suitable for clustering in high-dimensional embedding spaces where Euclidean distance metrics often perform poorly due to the curse of dimensionality.

**Algorithm Overview:**

The DP clustering algorithm in Clusterium follows these key steps:

1. **Embedding Generation**: Transform documents into normalized vector representations using a pretrained language model.

2. **Sequential Processing**: Documents are processed one at a time, following the Chinese Restaurant Process metaphor:

   - For each document, calculate the probability of joining each existing cluster or creating a new one
   - These probabilities combine the CRP prior with the von Mises-Fisher likelihood
   - Sample a cluster assignment based on these probabilities

3. **Cluster Maintenance**: When a document is assigned to a cluster, the cluster's parameters are updated to incorporate the new information.

4. **Inference**: For prediction on new documents, a deterministic approach is used by selecting the most likely cluster rather than sampling, enhancing stability.

A notable deviation from some classical implementations is the use of log-space calculations to prevent numerical underflow when dealing with high-dimensional embeddings, a critical consideration for text clustering applications.

**Key Design Considerations:**

Clusterium's implementation includes several important design decisions that affect clustering behavior:

1. **Directional Representation**: By focusing on normalized vectors, the algorithm emphasizes semantic direction rather than magnitude, which is particularly appropriate for text embeddings where relative word importance is encoded in the directional relationships.

2. **Stochastic vs. Deterministic Assignment**: During training, cluster assignments are sampled probabilistically to enable exploration of the clustering space, while during prediction, a deterministic maximum-likelihood approach is used to ensure consistency.

3. **Efficient Cluster Representation**: Each cluster is concisely represented by its mean direction and size, enabling efficient updates and likelihood calculations.

4. **Machine Learning Framework Integration**: The implementation follows established patterns from the machine learning ecosystem, ensuring interoperability with other tools and workflows.

**Stochastic Properties and Document Order Sensitivity:**

A critical aspect of the DP implementation is its sequential, stochastic nature. Since documents are processed one at a time following the Chinese Restaurant Process, several important properties emerge:

1. **Order Dependency**: The final clustering outcome is sensitive to the order in which documents are processed. This sensitivity arises because:

   - Early documents establish initial clusters that influence subsequent assignments
   - The rich-get-richer effect of the CRP amplifies small differences in initial conditions
   - The exploration-exploitation trade-off evolves as more documents are assigned

2. **Probabilistic Assignments**: Unlike deterministic algorithms like K-means, the DP model samples cluster assignments according to probabilities rather than deterministically choosing the closest centroid. This stochastic assignment:

   - Enables exploration of the clustering space
   - Helps prevent converging to poor local optima
   - Introduces variability in results between runs
   - Requires careful control through the random seed for reproducibility

3. **Convergence Properties**: Since the algorithm processes documents sequentially without iterations or global optimization, it does not "converge" in the traditional sense. Instead:

   - Each document is assigned once based on the current state
   - The final clustering depends on the complete sequence of decisions
   - Multiple runs with different random seeds can produce different valid clusterings

To mitigate order dependency in production applications, randomly shuffling documents before clustering is recommended—a practice implemented in Clusterium's test suite.

**Parameter Tuning:**

The DP clustering model is governed by two key parameters that significantly influence clustering behavior from an academic perspective:

1. **Alpha (α)**: The concentration parameter that controls cluster proliferation.

   - Low values (0.5-1.0) produce fewer, larger clusters focusing on major thematic distinctions
   - Medium values (1.0-5.0) generate moderate numbers of clusters capturing subtopic variations
   - High values (5.0-10.0) create numerous smaller clusters identifying fine-grained distinctions

2. **Kappa (κ)**: The precision parameter for the von Mises-Fisher distribution that determines cluster granularity.

   - Higher values (15.0-25.0) create tightly defined clusters with strict semantic boundaries
   - Moderate values (8.0-15.0) balance cohesion with reasonable cluster sizes
   - Lower values (5.0-8.0) yield more flexible cluster boundaries, accommodating greater variation

The interaction between these parameters creates distinct clustering profiles. For example, a combination of low α (1.0) with high :math:`\kappa` (20.0) tends to produce a small number of well-separated clusters corresponding to major conceptual categories, while higher α (5.0) with moderate :math:`\kappa` (10.0) reveals more fine-grained topic structure with hierarchical relationships between concepts.

Pitman-Yor Process Clustering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clustering using the Pitman-Yor Process (PYP) is generally better suited for text data as it can model the power-law distributions common in natural language.

.. note::

   This section is currently under development and will be added in a future update.

Evaluation methodology
----------------------

This section documents the methodological considerations behind the evaluation metrics implemented in Clusterium.

Silhouette Score Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When evaluating clustering results, especially those from Bayesian nonparametric models like the Pitman-Yor Process, singleton clusters (clusters with only 1 sample) are common and expected. The standard silhouette score calculation requires at least 2 samples per cluster, creating a methodological challenge.

The silhouette coefficient for a sample *i* is defined as:

.. math::

   s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}

where :math:`a(i)` is the mean distance between sample *i* and all other samples in the same cluster, and :math:`b(i)` is the mean distance to samples in the nearest neighboring cluster.

**Our approach:**

Rather than returning a zero score when any singleton clusters exist (which would effectively discard valuable information about well-formed clusters), Clusterium implements a more nuanced
approach that:

1. Identifies valid clusters (those with ≥2 samples)
2. Filters samples to include only those belonging to valid clusters
3. Calculates the silhouette score using only these valid samples and clusters

This methodology preserves information about cluster quality while respecting the mathematical requirements of the silhouette coefficient. The implementation logs detailed information about how many samples and clusters were included in the calculation, providing full transparency.

For the detailed implementation, see the ``calculate_silhouette_score`` method in the :class:`clusx.evaluation.ClusterEvaluator` class.

**Justification:**

This approach was chosen because it:

- Provides more informative evaluation results
- Better represents the quality of the valid portions of the clustering
- Aligns with established practices in cluster validation literature [3]_, where excluding invalid clusters in evaluation metrics is an accepted methodology
- Avoids misleading zero scores when meaningful clusters exist

**Example:**

Consider a clustering result with 10 samples and 3 clusters:

- Cluster 1: 2 samples
- Cluster 2: 1 sample (singleton)
- Cluster 3: 7 samples

Without filtering, the silhouette score would be zero because of the singleton cluster.

With filtering, Clusterium would:

1. Identify the 2 valid clusters (Cluster 1 and Cluster 3)
2. Filter the samples to include only those in valid clusters
3. Calculate the silhouette score using the filtered samples and clusters


References
----------

.. [1] Ferguson, T. S. (1973). "A Bayesian Analysis of Some Nonparametric Problems". The Annals of Statistics. 1(2): 209–230. doi:`10.1214/aos/1176342360 <https://doi.org/10.1214/aos/1176342360>`_

.. [2] Banerjee, A., Dhillon, I. S., Ghosh, J., & Sra, S. (2005). "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions". Journal of Machine Learning Research, 6, 1345-1382. https://dl.acm.org/doi/10.5555/1046920.1088718

.. [3] Rousseeuw, P. J. (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis". Journal of Computational and Applied Mathematics. 20: 53–65. doi:`10.1016/0377-0427(87)90125-7 <https://doi.org/10.1016/0377-0427(87)90125-7>`_
