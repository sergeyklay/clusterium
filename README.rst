==========
Clusterium
==========

|ci| |codecov| |docs|

.. -teaser-begin-

Clusterium is a Bayesian nonparametric toolkit for text clustering, analysis,
and benchmarking that leverages state-of-the-art embedding models and statistical
validation techniques.

.. -teaser-end-

.. -overview-begin-

Features
--------

- **Dirichlet Process Clustering**: Implements the Dirichlet Process for text clustering
- **Pitman-Yor Process Clustering**: Implements the Pitman-Yor Process for text clustering with improved performance
- **Evaluation**: Evaluates clustering results using a variety of metrics, including Silhouette Score, Davies-Bouldin Index, and Power-law Analysis
- **Visualization**: Generates plots of cluster size distributions

Quick Start
-----------

.. code-block:: bash

   pip install clusx

   # Run clustering with optimized default parameters
   clusx cluster --input your_data.txt --output clusters.csv

   # For custom parameter tuning, you can override the defaults
   clusx cluster --input your_data.txt \
     --dp-alpha 1.0 \
     --pyp-alpha 0.5 \
     --pyp-sigma 0.5 \
     --variance 0.2 \
     --output clusters.csv

   # Evaluate clustering results and generate visualizations
   clusx evaluate \
     --input your_data.txt \
     --dp-clusters output/clusters_output_dp.csv \
     --pyp-clusters output/clusters_output_pyp.csv \
     --plot

.. note::

   The default parameters are now optimized based on extensive testing:

   * Dirichlet Process: α=0.5, variance=0.3
   * Pitman-Yor Process: α=0.3, σ=0.3, variance=0.3

Python API Example
------------------

.. code-block:: python

   from clusx.clustering import DirichletProcess, PitmanYorProcess
   from clusx.clustering.utils import load_data, save_clusters_to_json

   # Load data
   texts = load_data("your_data.txt")
   # Or from CSV: texts = load_data("your_data.csv", column="your_column")

   # Perform Dirichlet Process clustering with default parameters
   base_measure = {"variance": 0.3}
   dp = DirichletProcess(alpha=0.5, base_measure=base_measure, random_state=42)
   clusters_dp, _ = dp.fit(texts)

   # Perform Pitman-Yor Process clustering with default parameters
   base_measure = {"variance": 0.3}  # Same variance for both models
   pyp = PitmanYorProcess(alpha=0.3, sigma=0.3, base_measure=base_measure, random_state=42)
   clusters_pyp, _ = pyp.fit(texts)

   # Save results
   save_clusters_to_json("dp_clusters.json", texts, clusters_dp, "DP")
   save_clusters_to_json("pyp_clusters.json", texts, clusters_pyp, "PYP")


.. note::

   For detailed installation instructions, please see the `Installation Guide <https://clusterium.readthedocs.io/en/latest/installation.html>`_.
   And for usage instructions, use cases, examples, and advanced configuration options, please see the `Usage Guide <https://clusterium.readthedocs.io/en/latest/usage.html>`_.

.. -overview-end-

.. -project-information-begin-

Project Information
===================

Clusterium is released under the `MIT License <https://choosealicense.com/licenses/mit/>`_,
its documentation lives at `Read the Docs <https://clusterium.readthedocs.io/>`_,
the code on `GitHub <https://github.com/sergeyklay/clusterium>`_,
and the latest release on `PyPI <https://pypi.org/project/clusterium/>`_.
It's rigorously tested on Python 3.11+.

If you'd like to contribute to Clusterium you're most welcome!

.. -project-information-end-

.. -support-begin-

Support
=======

Should you have any question, any remark, or if you find a bug, or if there is
something you can't do with the Clusterium, please
`open an issue <https://github.com/sergeyklay/clusterium/issues>`_.

.. -support-end-

.. |ci| image:: https://github.com/sergeyklay/clusterium/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/sergeyklay/clusterium/actions/workflows/ci.yml
   :alt: CI

.. |codecov| image:: https://codecov.io/gh/sergeyklay/clusterium/branch/main/graph/badge.svg?token=T5d9KTXtqP
   :target: https://codecov.io/gh/sergeyklay/clusterium
   :alt: Coverage

.. |docs| image:: https://readthedocs.org/projects/clusterium/badge/?version=latest
   :target: https://clusterium.readthedocs.io/en/latest/?badge=latest
   :alt: Docs
