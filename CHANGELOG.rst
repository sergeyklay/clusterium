Changelog
=========

This file contains a brief summary of new features and dependency changes or
releases, in reverse chronological order.


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
