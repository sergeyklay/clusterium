==========
Clusterium
==========

|ci| |codecov| |docs|

.. -teaser-begin-

Clusterium is a toolkit for clustering, analyzing, and benchmarking text data using state-of-the-art embedding models and clustering algorithms.

.. -teaser-end-

.. -overview-begin-

Features
========

- **Dirichlet Process Clustering**: Implements the Dirichlet Process for text clustering
- **Pitman-Yor Process Clustering**: Implements the Pitman-Yor Process for text clustering with improved performance
- **Evaluation**: Evaluates clustering results using a variety of metrics, including Silhouette Score, Davies-Bouldin Index, and Power-law Analysis
- **Visualization**: Generates plots of cluster size distributions

Quick Start
===========

.. code-block:: bash

   pip install clusx

   # Run clustering
   clusx --input your_data.csv --column your_column --output clusters.csv

   # Evaluate clustering results and generate visualizations
   clusx evaluate \
     --input input.csv \
     --column your_column \
     --dp-clusters output_dp.csv \
     --pyp-clusters output_pyp.csv \
     --plot

Python API Example
------------------

.. code-block:: python

   from clusx.clustering import DirichletProcess
   from clusx.clustering.utils import load_data_from_csv, save_clusters_to_json

   # Load data
   texts, data = load_data_from_csv("your_data.csv", column="your_column")

   # Perform clustering
   dp = DirichletProcess(alpha=1.0)
   clusters, params = dp.fit(texts)

   # Save results
   save_clusters_to_json("clusters.json", texts, clusters, "DP", data)


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
It’s rigorously tested on Python 3.11+.

If you'd like to contribute to Clusterium you're most welcome!

.. -project-information-end-

.. -support-begin-

Support
=======

Should you have any question, any remark, or if you find a bug, or if there is
something you can't do with the Clusterium,
`please open an issue <https://github.com/sergeyklay/clusterium>`_.

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
