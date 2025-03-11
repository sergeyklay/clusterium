============
Installation
============

Overview
========

The Clusterium project provides two main components:

* **Python package** ``clusx``: A comprehensive library that can be imported and used in your Python applications for text clustering and analysis.
* **Command-line utility** ``clusx``: A convenient command-line interface that provides direct access to the package's functionality without writing Python code.

Both components are installed simultaneously when following the instructions below, allowing you to choose the most appropriate interface for your specific needs.

Prerequisites
=============

Before installing ``clusx``, ensure you have the following prerequisites:

* Python 3.11 or higher
* `Poetry <https://python-poetry.org/>`_ for dependency management
* Git (for cloning the repository)

Python Version Compatibility
----------------------------

``clusx`` requires Python 3.11 or higher. This requirement ensures access to the latest language features and optimizations. The project is tested with Python 3.11, 3.12, and 3.13.

If you're using an older version of Python, you'll need to upgrade before installing ``clusx``:

.. code-block:: bash

   # Check your current Python version
   python --version

Project Structure
-----------------

The Clusterium project follows a standard Python package structure:

* ``clusx/``: Main package directory containing the source code
* ``tests/``: Test suite for verifying functionality
* ``docs/``: Documentation files (including this installation guide)
* ``pyproject.toml``: Project configuration and dependencies

Installation Steps
==================

Basic Installation
------------------

Follow these steps for a standard installation:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/sergeyklay/clusterium.git
      cd clusterium

2. Create and activate the virtual environment:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate

3. Install dependencies using Poetry:

   .. code-block:: bash

      poetry install

   This will install all core dependencies required for running the application.

Installation Options
--------------------

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

If you plan to contribute to the project or need development tools:

.. code-block:: bash

   poetry install --with dev,testing,docs

This installs all dependency groups, including development tools, testing frameworks, and documentation generators.

Production Installation
^^^^^^^^^^^^^^^^^^^^^^^

For production environments where you only need the core functionality:

.. code-block:: bash

   poetry install --without dev,testing,docs

This minimizes the installation footprint by excluding development-related dependencies.

Custom Installation
^^^^^^^^^^^^^^^^^^^

You can customize which dependency groups to include based on your specific needs. For example:

.. code-block:: bash

   # For development without documentation tools
   poetry install --with dev,testing --without docs

   # For documentation work only
   poetry install --with docs --without dev,testing

Verifying Installation
======================

To verify that the installation was successful, run:

.. code-block:: bash

   clusx --help

Or using the Python module:

.. code-block:: bash

   python -m clusx --help

You should see the help message with available command-line options.

Dependencies
============

The project uses Poetry for dependency management and organizes dependencies into several groups:

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

Poetry allows installing specific dependency groups based on your needs:

Development Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

Tools for development and code quality:

.. code-block:: bash

   poetry install --with dev

* ``black``: Code formatter
* ``debugpy``: Debugging tool
* ``flake8``: Linter
* ``isort``: Import sorter
* ``pre-commit``: Git hooks manager

Testing Dependencies
^^^^^^^^^^^^^^^^^^^^

Tools for testing the codebase:

.. code-block:: bash

   poetry install --with testing

* ``pytest``: Testing framework
* ``coverage``: Code coverage tool

Documentation Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

Tools for building documentation:

.. code-block:: bash

   poetry install --with docs

* ``sphinx``: Documentation generator
* ``sphinx-rtd-theme``: Read the Docs theme for Sphinx

Installing Multiple Groups
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can install multiple dependency groups at once:

.. code-block:: bash

   poetry install --with dev,testing,docs

Installing Only Core Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you only need the core functionality without any development tools:

.. code-block:: bash

   poetry install --without dev,testing,docs

These dependencies will be automatically installed by Poetry based on the options you choose.

Troubleshooting
===============

Common Issues
-------------

If you encounter any issues during installation:

1. Ensure you have the correct Python version (3.11+)
2. Make sure Poetry is `properly installed <https://python-poetry.org/docs/#installing-with-the-official-installer>`_
3. Check for any error messages during the installation process

PyTorch Installation Issues
---------------------------

If you encounter issues with PyTorch installation:

.. code-block:: bash

   # Install PyTorch separately with CUDA support if needed
   pip install torch --index-url https://download.pytorch.org/whl/cu118

   # Then continue with Poetry installation
   poetry install --no-dev

Dependency Conflicts
--------------------

If you encounter dependency conflicts:

.. code-block:: bash

   # Update Poetry
   poetry self update

   # Clear Poetry's cache
   poetry cache clear --all pypi

   # Try installation with verbose output
   poetry install -v

Virtual Environment Issues
--------------------------

If you have issues with the virtual environment:

.. code-block:: bash

   # Create a fresh virtual environment
   rm -rf .venv
   python -m venv .venv
   source .venv/bin/activate

   # Install Poetry in the virtual environment
   pip install poetry
   poetry install

Getting Help
------------

For more detailed help, please open an issue on the `GitHub repository <https://github.com/sergeyklay/clusterium/issues>`_.
