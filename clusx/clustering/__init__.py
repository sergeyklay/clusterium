"""
Clustering module for Bayesian nonparametric text analysis.

This module provides implementations of Dirichlet Process and Pitman-Yor Process
for clustering text data using Bayesian nonparametric methods. These algorithms
enable flexible, data-driven clustering without requiring a predefined number of
clusters.

Key components:

- DirichletProcess: Implementation of the Chinese Restaurant Process formulation
  of Dirichlet Process clustering
- PitmanYorProcess: Extension of Dirichlet Process that incorporates a discount
  parameter to model power-law distributions

Both models use sentence embeddings to represent text in a high-dimensional space
and perform clustering based on semantic similarity.

Typical usage:

    >>> from clusx.clustering import DirichletProcess
    >>> dp = DirichletProcess(alpha=0.5, base_measure={"variance": 0.3})
    >>> texts = ["text1", "text2", "text3", "..."]
    >>> clusters, params = dp.fit(texts)
"""

from .models import DirichletProcess, PitmanYorProcess

__all__ = ["DirichletProcess", "PitmanYorProcess"]
