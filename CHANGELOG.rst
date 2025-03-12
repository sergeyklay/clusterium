Changelog
=========

This file contains a brief summary of new features and dependency changes or
releases, in reverse chronological order.


0.4.0 - 2025-XX-XX
------------------

Improved Documentation
^^^^^^^^^^^^^^^^^^^^^^

* Amend project documentation.

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

* Implement Proper Bayesian Inference: Implements log CRP/PYP priors and
  Gaussian likelihoods instead of heuristic similarity scoring.
  Fixes incorrect probabilistic model through valid posterior sampling.
* PYP Initialization: Properly initializes cluster parameters via parent class.
  Fixes PYP initialization bug.

Improvements
^^^^^^^^^^^^

* Embedding Efficiency: Precomputes and caches all embeddings upfront (``text_embeddings`` dict).
  Fixes O(N²) embedding calls.
* Reproducibility: Add ``random_state`` for controlled sampling via ``np.random.RandomState``.
  Addresses non-determinism.

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
