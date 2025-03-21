============
Installation
============

Overview
========

The Clusterium project provides two main components:

* **Python package** ``clusx``: A comprehensive library that can be imported and used in your Python applications for text clustering and analysis.
* **Command-line utility** ``clusx``: A convenient command-line interface that provides direct access to the package's functionality without writing Python code.

Both components are installed simultaneously when following the instructions below, allowing you to choose the most appropriate interface for your specific needs.

Requirements
============

Before installing ``clusx``, ensure you have the following prerequisites:

* Python 3.11 or higher
* `pip <https://pip.pypa.io/en/stable/>`_ (for PyPI installation)
* `Poetry <https://python-poetry.org/>`_ 2.0 or higher (for development installation only)
* `Git <https://git-scm.com/>`_ (for cloning the repository)

Python Version Compatibility
----------------------------

``clusx`` requires Python 3.11 or higher. This requirement ensures access to the latest language features and optimizations. The project is tested with Python 3.11, 3.12, and 3.13.

If you're using an older version of Python, you'll need to upgrade before installing ``clusx``:

.. code-block:: bash

   # Check your current Python version
   python --version

.. warning::
   **Python Command Differences Across Operating Systems**

   The ``python`` and ``pip`` commands may behave differently across operating systems:

   * **macOS**: By default, ``python`` often points to Python 2.x. You should use ``python3`` and ``pip3`` commands instead.
   * **Linux**: Many distributions now default ``python`` to Python 3.x, but some still maintain ``python`` as Python 2.x. Use ``python3`` and ``pip3`` to ensure you're using the correct version.
   * **Windows**: Recent installations typically have ``python`` pointing to Python 3.x, but it's best to verify with ``python --version``.

   To ensure you're using the correct Python version, always check:

   .. code-block:: bash

      # Check Python version
      python --version  # or python3 --version

      # Check pip version
      pip --version  # or pip3 --version

   If ``python`` points to Python 2.x on your system, replace all ``python`` commands in this guide with ``python3`` and all ``pip`` commands with ``pip3``.

   **Finding Your Python Installations**

   To check where Python is installed and how many Python versions you have on your system:

   **On Unix-based systems (Linux, macOS):**

   .. code-block:: bash

      # Find the location of the python/python3 executable
      which python
      which python3

      # Alternative command to find executable location
      command -v python
      command -v python3

      # List all instances of python in your PATH
      type -a python
      type -a python3

      # Check if you have multiple Python installations
      ls -l /usr/bin/python*
      ls -l /usr/local/bin/python*

   **On Windows:**

   .. code-block:: bash

      # Find the location of Python executable
      where python
      where python3

      # Check Python version and installation path
      py -0

Installation Methods
====================

There are several ways to install ``clusx`` depending on your needs:

Installing from PyPI (Recommended)
----------------------------------

``clusx`` is a Python package `hosted on PyPI <https://pypi.org/project/clusx/>`_.
The recommended installation method is using `pip <https://pip.pypa.io/en/stable/>`_ to install into a virtual environment:

.. code-block:: bash

   # Create and activate a virtual environment (recommended)
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install clusx
   python -m pip install clusx

   # Alternative commands if python points to Python 2.x on your system
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   python3 -m pip install clusx
   # or
   pip3 install clusx

After installation, the ``clusx`` command will be available from the command line:

.. code-block:: bash

   # Verify installation
   clusx --version

More information about ``pip`` and PyPI can be found here:

* `Install pip <https://pip.pypa.io/en/latest/installation/>`_
* `Python Packaging User Guide <https://packaging.python.org/>`_

Installing from GitHub Releases
-------------------------------

Another way to install package is to download it from GitHub Releases page:

1. Visit the `GitHub Releases page <https://github.com/sergeyklay/clusterium/releases>`_
2. Download the desired release artifacts (both ``.whl`` and/or ``.tar.gz`` files)
3. Download the corresponding checksum files (``SHA256SUMS``, ``SHA512SUMS``, or ``MD5SUMS``)
4. Verify the integrity of the downloaded files:

   .. code-block:: bash

      # Verify with SHA256 (recommended)
      sha256sum -c SHA256SUMS

5. Install the verified package:

   .. code-block:: bash

      # Create a directory for the download
      mkdir clusx-download && cd clusx-download

      # Download the latest release artifacts and checksums (replace X.Y.Z with the actual version)
      # You can use wget or curl
      wget https://github.com/sergeyklay/clusterium/releases/download/X.Y.Z/clusx-X.Y.Z-py3-none-any.whl
      wget https://github.com/sergeyklay/clusterium/releases/download/X.Y.Z/clusx-X.Y.Z.tar.gz
      wget https://github.com/sergeyklay/clusterium/releases/download/X.Y.Z/SHA256SUMS

      # Verify the integrity of the downloaded files
      sha256sum -c SHA256SUMS

      # Create and activate a virtual environment
      python -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

      # Install the verified package (choose one)
      pip install clusx-X.Y.Z-py3-none-any.whl  # Wheel file (recommended)
      # OR
      pip install clusx-X.Y.Z.tar.gz  # Source distribution

      # If python points to Python 2.x on your system
      pip3 install clusx-X.Y.Z-py3-none-any.whl
      # Or
      pip3 install clusx-X.Y.Z.tar.gz

      # Verify the installation
      clusx --version

Installing the Development Version
----------------------------------

If you need the latest unreleased features, you can install directly from the GitHub repository:

.. code-block:: bash

   # Install the latest development version
   python -m pip install -e git+https://github.com/sergeyklay/clusterium.git#egg=clusx

   # If python points to Python 2.x on your system
   python3 -m pip install -e git+https://github.com/sergeyklay/clusterium.git#egg=clusx

.. note::
   The ``main`` branch will always contain the latest unstable version, so the experience
   might not be as smooth. If you wish to use a stable version, consider installing from PyPI
   or switching to a specific `tag <https://github.com/sergeyklay/clusterium/tags>`_.

Installing for Development
--------------------------

If you plan to contribute to the project or need to modify the code, follow these steps:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/sergeyklay/clusterium.git
      cd clusterium

2. Create and activate a virtual environment:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

      # If python points to Python 2.x on your system
      python3 -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install with Poetry:

   .. code-block:: bash

      # Install Poetry if you haven't already
      # See https://python-poetry.org/docs/#installation

      # Install dependencies
      poetry install

Installation Options with Poetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Poetry allows for flexible installation options based on your specific needs:

**Full Development Environment**

To install all dependency groups, including development tools, testing frameworks, and documentation generators:

.. code-block:: bash

   poetry install --with dev,testing,docs

**Production Installation**

For production environments where you only need the core functionality:

.. code-block:: bash

   poetry install --without dev,testing,docs

**Custom Installation**

You can customize which dependency groups to include:

.. code-block:: bash

   # For development without documentation tools
   poetry install --with dev,testing --without docs

   # For documentation work only
   poetry install --with docs --without dev,testing

Verifying Installation
======================

To verify that the installation was successful, run:

.. code-block:: bash

   clusx --version

Or using the Python module:

.. code-block:: bash

   python -m clusx --version
   # If python points to Python 2.x on your system
   python3 -m clusx --version

You should see the version information and a brief copyright notice.

Dependencies
============

Core Dependencies
-----------------

These dependencies are installed by default and are required for the basic functionality:

* ``numpy``: For numerical operations
* ``sentence-transformers``: For text embeddings
* ``scipy``: For distance calculations
* ``matplotlib``: For visualization
* ``torch``: For deep learning operations
* ``tqdm``: For progress bars
* ``click``: For command-line interface
* ``pandas``: For data manipulation
* ``powerlaw``: For statistical analysis
* ``scikit-learn``: For machine learning algorithms

Optional Dependency Groups
--------------------------

When installing with Poetry, you can choose specific dependency groups:

Development Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

Tools for development and code quality:

* ``black``: Code formatter
* ``debugpy``: Debugging tool
* ``flake8``: Linter
* ``isort``: Import sorter
* ``pre-commit``: Git hooks manager

Testing Dependencies
^^^^^^^^^^^^^^^^^^^^

Tools for testing the codebase:

* ``pytest``: Testing framework
* ``coverage``: Code coverage tool

Documentation Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

Tools for building documentation:

* ``sphinx``: Documentation generator
* ``sphinx-rtd-theme``: Read the Docs theme for Sphinx

Troubleshooting
===============

Common Issues
-------------

If you encounter any issues during installation:

1. Ensure you have the correct Python version (3.11+)
2. Make sure you're using the latest version of pip or Poetry
3. Check for any error messages during the installation process

PyTorch Installation Issues
---------------------------

If you encounter issues with PyTorch installation:

.. code-block:: bash

   # Install PyTorch separately with CUDA support if needed
   pip install torch --index-url https://download.pytorch.org/whl/cu118

   # Then continue with the installation
   pip install clusx

Dependency Conflicts
--------------------

If you encounter dependency conflicts:

.. code-block:: bash

   # For pip installations, try:
   pip install --upgrade pip
   pip install clusx --no-deps
   pip install -r <(pip freeze | grep -v clusx)

   # For Poetry installations:
   poetry self update
   poetry lock --no-update
   poetry install
