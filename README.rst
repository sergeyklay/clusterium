==========
Clusterium
==========

|ci| |codecov| |docs|

.. -teaser-begin-

Clusterium is a Bayesian nonparametric toolkit for text clustering, analysis, and benchmarking that leverages state-of-the-art embedding models and statistical validation techniques.

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

   # Install the package
   pip install clusx

   # Basic clustering with default parameters
   clusx cluster --input your_data.txt

   # Evaluate clustering results
   clusx evaluate \
     --input your_data.txt \
     --dp-clusters output/clusters_output_dp.csv \
     --pyp-clusters output/clusters_output_pyp.csv

That's it! The tool uses optimized default parameters and saves all outputs to the ``output`` directory.

For interactive visualization during evaluation, add the ``--show-plot`` option:

.. code-block:: bash

   clusx evaluate \
     --input your_data.txt \
     --dp-clusters output/clusters_output_dp.csv \
     --pyp-clusters output/clusters_output_pyp.csv \
     --show-plot

.. note::

   The default parameters are optimized based on extensive testing:

   * Dirichlet Process: α=0.5, variance=0.3
   * Pitman-Yor Process: α=0.3, σ=0.3, variance=0.3

   For advanced usage and parameter tuning, see the `Usage Guide <https://clusterium.readthedocs.io/en/latest/usage.html>`_.

Python API Example
------------------

.. code-block:: python

   from clusx.clustering import DirichletProcess, PitmanYorProcess
   from clusx.clustering.utils import load_data

   # Load data
   texts = load_data("your_data.txt")

   # Perform clustering with default parameters
   dp = DirichletProcess(alpha=0.5)  # Dirichlet Process
   clusters_dp, _ = dp.fit(texts)

   pyp = PitmanYorProcess(alpha=0.3, sigma=0.3)  # Pitman-Yor Process
   clusters_pyp, _ = pyp.fit(texts)

   # Print number of clusters found
   print(f"DP found {len(set(clusters_dp))} clusters")
   print(f"PYP found {len(set(clusters_pyp))} clusters")

For more advanced usage, including saving results and evaluation, see the `Usage Guide <https://clusterium.readthedocs.io/en/latest/usage.html>`_.

.. note::

   For detailed installation instructions, please see the `Installation Guide <https://clusterium.readthedocs.io/en/latest/installation.html>`_. And for usage instructions, use cases, examples, and advanced configuration options, please see the `Usage Guide <https://clusterium.readthedocs.io/en/latest/usage.html>`_.

.. -overview-end-

.. -project-information-begin-

Project Information
===================

Clusterium is released under the `MIT License <https://choosealicense.com/licenses/mit/>`_, its documentation lives at `Read the Docs <https://clusterium.readthedocs.io/>`_, the code on `GitHub <https://github.com/sergeyklay/clusterium>`_, and the latest release on `PyPI <https://pypi.org/project/clusterium/>`_. It's rigorously tested on Python 3.11+.

If you'd like to contribute to Clusterium you're most welcome!

.. -project-information-end-

.. -support-begin-

Support
=======

Should you have any question, any remark, or if you find a bug, or if there is something you can't do with the Clusterium, please `open an issue <https://github.com/sergeyklay/clusterium/issues>`_.

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
