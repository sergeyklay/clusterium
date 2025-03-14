=========================
Methodological Framework
=========================

This document describes the methodological decisions and theoretical considerations underlying the Clusterium implementation. It explains how key algorithms are implemented, why specific  approaches were chosen, and their academic foundations.

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
- Aligns with established practices in cluster validation literature [1]_, where excluding invalid clusters in evaluation metrics is an accepted methodology
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

.. [1] Rousseeuw, P. J. (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis". Journal of Computational and Applied Mathematics. 20: 53–65. doi:`10.1016/0377-0427(87)90125-7 <https://doi.org/10.1016/0377-0427(87)90125-7>`_
