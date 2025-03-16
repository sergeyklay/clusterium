Changelog
=========

This file contains a brief summary of new features and dependency changes or
releases, in reverse chronological order.

0.7.0 - 2025-03-xx
------------------

Trivial/Internal Changes
^^^^^^^^^^^^^^^^^^^^^^^^

* Refactor type hints to use ``NDArray`` instead of ``EmbeddingTensor`` alias for better clarity and consistency in the ``DirichletProcess`` and ``PitmanYorProcess`` classes.
* Update logging for ``DirichletProcess`` and ``PitmanYorProcess`` to use the standard :mod:`logging` module.

0.6.0 - 2025-03-16
------------------

Features
^^^^^^^^

* Implemented von Mises-Fisher distribution for text embeddings, replacing Gaussian likelihood for better directional similarity
* Improved cluster initialization and updates with proper normalization techniques
* Added ``kappa`` parameter for explicit control over cluster cohesion
* Implemented global mean embedding as base measure for new clusters in CRP models


Breaking Changes
^^^^^^^^^^^^^^^^^

* Completely redesigned Dirichlet Process and Pitman-Yor Process implementations with incompatible APIs
* Removed ``variance`` parameter, replaced with more theoretically sound ``kappa`` parameter
* Changed cluster assignment methodology to use normalized embeddings and proper directional statistics
* Modified method signatures across clustering models for better scikit-learn compatibility


Bug Fixes
^^^^^^^^^

* Fixed incorrect Gaussian likelihood calculation that caused bias against new clusters
* Resolved numerical stability issues by implementing consistent log-space calculations
* Fixed "singleton cluster dominance" issue with proper cluster mean initialization
* Corrected PYP prior calculation to handle edge cases with small clusters appropriately


Improvements
^^^^^^^^^^^^

* Rewritten clustering core algorithm to properly handle directional text embeddings on the unit hypersphere
* Optimized embedding processing with efficient normalization and similarity calculations
* Enhanced API with scikit-learn compatible ``fit()``, ``predict()``, and ``fit_predict()`` methods
* Improved theoretical soundness with proper Bayesian inference for cluster assignments


Improved Documentation
^^^^^^^^^^^^^^^^^^^^^^

* Enhanced methodological framework with academic foundations and mathematical rigor
* Added detailed parameter tuning guidelines with practical ranges

0.5.0 - 2025-03-15
------------------

Features
^^^^^^^^

* Implement ``to_numpy`` helper function to convert PyTorch tensors to NumPy arrays.
* Add ``ClusterIntegrityError``, ``MissingClusterColumnError``, and ``MissingParametersError`` for better error handling.
* Enhance plotting functions with error handling:

  - Handle visualization-specific errors and properly report them.
  - Implemented ``safe_plot`` decorator for error handling in plotting functions.
  - Updated plotting functions to raise ``VisualizationError`` for missing or invalid data.
  - Improved documentation for new functionalities and added examples.
  - Removed deprecated plotting functions and streamlined visualization dashboard code.

Breaking Changes
^^^^^^^^^^^^^^^^^

* Refactored ``load_cluster_assignments`` function:

  - Now raises specific custom exceptions (``MissingClusterColumnError`` and ``MissingParametersError``) instead of generic ``ValueError``
  - Requires all parameters (alpha, sigma, variance) to be present in the CSV file
  - Removed fallback mechanism to extract parameters from filename
  - More specific cluster column detection (looking for ``cluster_`` prefix)
  - Improved docstring with better description of function behavior and exceptions

Bug Fixes
^^^^^^^^^

* Fix critical issues in similarity metrics calculation:

  - Properly handle singleton clusters instead of reporting misleading 0.0 values
  - Optimize computation for large datasets with sparse clusters
  - Add robust handling for edge cases with no valid cluster pairs
  - Implement correct averaging when mixing singleton and non-singleton clusters
  - Fix silent failures on datasets with predominantly singleton clusters

Improvements
^^^^^^^^^^^^

* Select appropriate colormaps based on visualization best practices for clustering.
* Redesign progress bar on clustering to be more informative and less noisy.
* Enhance silhouette score calculation to handle singleton clusters properly:

  - Now calculates scores using only valid clusters (≥2 samples) rather than returning 0.0 when any singleton exists
  - Preserves valuable evaluation data that would otherwise be discarded
  - Provides detailed logging about what proportion of data contributed to the score
  - Aligns with academic best practices in cluster validation literature

Improved Documentation
^^^^^^^^^^^^^^^^^^^^^^

* Fix code smells and style issues.
* Introduced ``pylint`` to the CI workflow.
* Added new "Methodological Framework" documentation explaining theoretical decisions behind implementation choices.

Trivial/Internal Changes
^^^^^^^^^^^^^^^^^^^^^^^^

* Amend and improve installation documentation.

0.4.0 - 2025-03-13
------------------

Features
^^^^^^^^

* Updated the application interface to support both text files (each line treated as a clustering candidate) and CSV files.
* Added ``--show-plot/--no-show-plot`` option to the ``evaluate`` command to control whether plots are displayed interactively. Default is ``--no-show-plot`` to better support automation and headless environments.

Breaking Changes
^^^^^^^^^^^^^^^^

* Removed the "answer" field from ``*_dp.json`` and ``*_pyp.json`` outputs, with corresponding updates to code, documentation, and tests.
* CSV inputs now require an explicit column name; otherwise, the program will exit with an error.
* Changed default parameter values to optimal settings:

  - Dirichlet Process: α=0.5 (was 5.0)
  - Pitman-Yor Process: α=0.3 (was 5.0), σ=0.3 (was 0.5)
  - Variance: 0.3 (was 0.1)

Bug Fixes
^^^^^^^^^

* Fixed critical parameter handling in CLI interface for Dirichlet Process and Pitman-Yor Process:

  - Separated ``--dp-alpha`` and ``--pyp-alpha`` parameters with appropriate help text
  - Added proper validation for parameter ranges (DP: α > 0, PYP: α > -σ, 0 ≤ σ < 1)
  - Updated documentation to clarify that using the same α value for both models leads to dramatically different clustering behaviors
  - Added recommended parameter ranges in help text (DP: α ∈ [0.1, 5.0], PYP: α ∈ [0.1, 2.0], σ ∈ [0.1, 0.7])

Improvements
^^^^^^^^^^^^

* The resulted JSON output file no longer created as it was identical to the Dirichlet Process JSON output file.
* Default parameter values now set to optimal values based on extensive testing, providing better out-of-the-box clustering performance.
* Improved visualization handling with non-interactive plot generation by default, making the tool more suitable for automated pipelines and CI/CD environments.

Improved Documentation
^^^^^^^^^^^^^^^^^^^^^^

* Amend and improve usage documentation.
* Amend and improve API documentation.
* Updated documentation to reflect new default parameter values and their effects on clustering.
* Enhanced documentation with clear examples of interactive vs. non-interactive visualization options in both CLI and Python API.

Trivial/Internal Changes
^^^^^^^^^^^^^^^^^^^^^^^^

* Improve cascading metadata resolution in :mod:`clusx.version` module.
* Refactor type hints to use built-in types.
* Remove embedding cache functionality as it is not helpful for the current implementation. It will be re-implemented in the future.

0.3.3 - 2025-03-12
------------------

Trivial/Internal Changes
^^^^^^^^^^^^^^^^^^^^^^^^

* Fix CD workflow with release artifact upload.

0.3.2 - 2025-03-12
------------------

Improved Documentation
^^^^^^^^^^^^^^^^^^^^^^

* Amend project documentation.

Trivial/Internal Changes
^^^^^^^^^^^^^^^^^^^^^^^^

* Add checksum generation and verification to CD workflow.

0.3.1 - 2025-03-12
------------------

Trivial/Internal Changes
^^^^^^^^^^^^^^^^^^^^^^^^

* Fix publishing to PyPI.

0.3.0 - 2025-03-12
------------------

Bug Fixes
^^^^^^^^^

* Implement Proper Bayesian Inference: Implements log CRP/PYP priors and Gaussian likelihoods instead of heuristic similarity scoring. Fixes incorrect probabilistic model through valid posterior sampling.
* PYP Initialization: Properly initializes cluster parameters via parent class. Fixes PYP initialization bug.

Improvements
^^^^^^^^^^^^

* Embedding Efficiency: Precomputes and caches all embeddings upfront (``text_embeddings`` dict). Fixes O(N²) embedding calls.
* Reproducibility: Add ``random_state`` for controlled sampling via ``np.random.RandomState``. Addresses non-determinism.

Trivial/Internal Changes
^^^^^^^^^^^^^^^^^^^^^^^^

* Change project name.

Improved Documentation
^^^^^^^^^^^^^^^^^^^^^^

* Add initial project documentation.

0.2.0 - 2025-03-11
------------------

Features
^^^^^^^^

* Migrate to Dirichlet & Pitman-Yor Process.
* Add comprehensive evaluation dashboard and power-law analysis.
* Add integration and unit tests for clustering models.

Breaking Changes
^^^^^^^^^^^^^^^^

* Drop support for DBSCAN clustering.
* Drop support for custom embedding model.

0.1.0 - 2025-03-10
------------------

* Initial release.
