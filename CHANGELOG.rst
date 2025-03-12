Changelog
=========

This file contains a brief summary of new features and dependency changes or
releases, in reverse chronological order.


0.4.0 - 2025-XX-XX
------------------

Features
^^^^^^^^

* Updated the application interface to support both text files (each line treated
  as a clustering candidate) and CSV files.

Breaking Changes
^^^^^^^^^^^^^^^^

* Removed the "answer" field from ``*_dp.json`` and ``*_pyp.json`` outputs, with
  corresponding updates to code, documentation, and tests.
* CSV inputs now require an explicit column name; otherwise, the program will
  exit with an error.

Improvements
^^^^^^^^^^^^

* The resulted JSON output file no longer created as it was identical to the
  Dirichlet Process JSON output file.

Improved Documentation
^^^^^^^^^^^^^^^^^^^^^^

* Amend and improve usage documentation.

Trivial/Internal Changes
^^^^^^^^^^^^^^^^^^^^^^^^

* Improve cascading metadata resolution in :mod:`clusx.version` module.
* Refactor type hints to use built-in types.
* Remove embedding cache functionality as it is not helpful for the current
  implementation. It will be re-implemented in the future.

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
