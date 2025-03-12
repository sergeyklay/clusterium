==================
Maintainers' Guide
==================

This document outlines essential guidelines for maintaining the Clusterium project.
It provides instructions for testing, building, and deploying the package, as well
as managing CI workflows.

Overview
========

The Clusterium project is managed via Poetry and adheres to modern Python packaging
standards. This guide assumes familiarity with GitHub Actions, Poetry, and common Python
development workflows.

Key configurations:

* **Python Versions Supported:** >= 3.11 (tested on 3.11, 3.12, and 3.13)
* **Build Tool:** ``poetry``
* **Primary Dependencies:** ``numpy``, ``sentence-transformers``, ``scipy``, ``matplotlib``
* **Documentation Tool:** ``sphinx`` with Read the Docs theme
* **Testing Tools:** ``pytest``, ``coverage``
* **Linting Tools:** ``black``, ``flake8``, ``isort``

.. note::

   While the project provides a Makefile to simplify common development tasks,
   all operations can also be performed using direct commands without requiring
   the ``make`` program. This guide includes both Makefile commands and their
   direct command equivalents.

Development Environment
=======================

The project provides a Makefile to simplify common development tasks.

Prerequisites
-------------

To use the provided Makefile commands, you need to have the ``make`` program installed on your system:

* **Linux/macOS**: Usually pre-installed or available through package managers (``apt-get install make``, ``brew install make``)
* **Windows**: Available through tools like MSYS2, MinGW, Cygwin, or Windows Subsystem for Linux (WSL)

If you don't have ``make`` installed or prefer not to use it, equivalent direct commands are provided throughout this guide.

Setting Up
----------

Clone the repository and install dependencies:

.. code-block:: bash

   git clone https://github.com/sergeyklay/clusterium.git
   cd clusterium
   make install

If you don't have ``make`` installed, you can use the equivalent Poetry command:

.. code-block:: bash

   git clone https://github.com/sergeyklay/clusterium.git
   cd clusterium
   poetry install

This will install the package and all its dependencies using Poetry.

Available Make Commands
-----------------------

The project includes several make commands to streamline development:

.. code-block:: bash

   make help              # Show available commands and environment information
   make install           # Install the package and dependencies
   make test              # Run tests
   make ccov              # Generate combined coverage reports
   make format            # Format code using black and isort
   make format-check      # Check code formatting
   make lint              # Run linters
   make docs              # Test and build documentation
   make clean             # Remove build artifacts and directories

For each make command, equivalent direct commands are provided in the relevant sections below.

Testing the Project
===================

Unit tests and coverage reporting are managed using ``pytest`` and ``coverage``.

Running Tests Locally
---------------------

Run tests using the make command:

.. code-block:: bash

   make test

Or manually with Poetry:

.. code-block:: bash

   poetry run coverage erase
   poetry run coverage run -m pytest ./clusx ./tests
   poetry run coverage combine
   poetry run coverage report

Generate Coverage Reports
-------------------------

Generate HTML, XML, and LCOV coverage reports:

.. code-block:: bash

   make ccov

This will create reports in the ``coverage/`` directory with subdirectories for each format.

Without ``make``, use these Poetry commands:

.. code-block:: bash

   mkdir -p coverage/html coverage/xml coverage/lcov
   poetry run coverage combine || true
   poetry run coverage report
   poetry run coverage html -d coverage/html
   poetry run coverage xml -o coverage/xml/coverage.xml

CI Workflow
-----------

Tests are executed automatically on supported platforms and Python versions (3.11, 3.12, and 3.13) on Ubuntu. See the configuration in ``.github/workflows/ci.yml``.

The CI workflow includes:

* Code formatting verification
* Linting checks
* Unit tests with coverage reporting
* Coverage report upload to Codecov

Building the Package
====================

The ``clusx`` package is distributed in ``wheel`` and ``sdist`` formats.

Local Build
-----------

Install build dependencies:

.. code-block:: bash

   poetry install

Build the package:

.. code-block:: bash

   poetry build

Verify the built package:

.. code-block:: bash

   pip install dist/*.whl
   clusx --help

CI Workflow
-----------

The build workflow in ``.github/workflows/cd.yml`` ensures the package is built and verified across multiple Python versions.

Documentation Management
========================

Documentation is written using ``sphinx`` with the Read the Docs theme.

Building Documentation Locally
------------------------------

Install documentation dependencies:

.. code-block:: bash

   poetry install --with docs

Build the documentation using the ``Makefile`` from the root directory:

.. code-block:: bash

   make docs

Or build directly with sphinx:

.. code-block:: bash

   # Test documentation files
   python -m doctest CONTRIBUTING.rst README.rst

   # Build HTML documentation
   python -m sphinx \
      --jobs auto \
      --builder html \
      --nitpicky \
      --show-traceback \
      --fail-on-warning \
      --doctree-dir docs/build/doctrees \
      docs/source docs/build/html

View the documentation:

.. code-block:: bash

   # On Linux/macOS
   open docs/build/html/index.html

   # On Windows
   start docs/build/html/index.html

Other Documentation Formats
---------------------------

The docs ``Makefile`` supports various output formats:

.. code-block:: bash

   cd docs
   make epub      # Build EPUB documentation
   make man       # Build man pages
   make clean     # Clean build directory

Without ``make``, use these sphinx-build commands:

.. code-block:: bash

   cd docs

   # Build EPUB documentation
   sphinx-build -b epub source build/epub

   # Build man pages
   sphinx-build -b man source build/man

   # Clean build directory
   rm -rf build/

CI Workflow
-----------

The docs workflow automatically builds and validates documentation on pushes and pull requests. See ``.github/workflows/docs.yml``.

Linting and Code Quality Checks
===============================

Code quality is enforced using ``black``, ``flake8``, and ``isort``.

Running Locally
---------------

Format code and run linters using make commands:

.. code-block:: bash

   make format       # Format code with black and isort
   make format-check # Check formatting without making changes
   make lint         # Run flake8

Or manually with Poetry:

.. code-block:: bash

   # Format code (equivalent to make format)
   poetry run isort --profile black --python-version auto ./
   poetry run black . ./clusx ./tests

   # Check formatting without changes (equivalent to make format-check)
   poetry run isort --check-only --profile black --python-version auto --diff ./
   poetry run black --check . ./clusx ./tests

   # Run linters (equivalent to make lint)
   poetry run flake8 ./

Pre-commit Hooks
----------------

The project uses pre-commit hooks to ensure code quality before commits:

.. code-block:: bash

   # Install pre-commit hooks
   pre-commit install

   # Run pre-commit hooks on all files
   pre-commit run --all-files

CI Workflow
-----------

The CI workflow in ``.github/workflows/ci.yml`` includes formatting and linting checks. Pull requests with formatting issues will show the diff of improperly formatted files.

Release Process
===============

The release process involves version tagging and package publishing to PyPI.

Steps for Release
-----------------

1. Ensure all tests pass and documentation builds successfully
2. Update ``CHANGELOG.md`` with the changes in the new version
3. Tag the version using git and push tag to GitHub:

   .. code-block:: bash

      git tag -a v0.x.y -m "Release v0.x.y"
      git push origin v0.x.y

4. Build and publish the package:

   .. code-block:: bash

      poetry build
      poetry publish

CI Workflow
-----------

The release workflow is triggered when a new tag matching the pattern ``v*`` is pushed to GitHub. It builds the package and publishes it to PyPI.

Continuous Integration and Deployment
=====================================

CI/CD is managed via GitHub Actions, with workflows for:

* **Testing:** Ensures functionality and compatibility across Python 3.11, 3.12, and 3.13 on Ubuntu
* **Linting:** Maintains code quality with flake8, black, and isort
* **Documentation:** Validates and builds project documentation
* **Building:** Verifies the package's integrity
* **Release:** Publishes the package to PyPI

The CI workflow includes:

* Caching of dependencies to speed up builds
* Automatic code formatting verification
* Coverage reporting to Codecov
* JUnit XML test results

Development Guidelines
======================

Code Style
----------

The project follows the Black code style. Configuration is in ``pyproject.toml``:

.. code-block:: toml

   [tool.black]
   line-length = 88
   target-version = ["py312"]

Import Sorting
--------------

Imports should be sorted using isort with the Black profile:

.. code-block:: toml

   [tool.isort]
   profile = "black"
   py_version = 312

Type Annotations
----------------

Use type annotations for all function parameters and return values:

.. code-block:: python

   def process_text(text: str, threshold: float = 0.5) -> list[str]:
       """Process the input text and return a list of tokens."""
       # Implementation

Documentation Standards
-----------------------

* Use Google-style docstrings for all public functions, classes, and methods
* Include examples in docstrings where appropriate
* Keep the documentation up-to-date with code changes

Example docstring:

.. code-block:: python

   def calculate_similarity(text1: str, text2: str) -> float:
       """Calculate the semantic similarity between two texts.

       Args:
           text1: The first text string
           text2: The second text string

       Returns:
           A float between 0 and 1 representing similarity

       Example:
           >>> calculate_similarity("Hello world", "Hi world")
           0.85
       """
       # Implementation

Troubleshooting
===============

Common Development Issues
-------------------------

1. **Poetry environment issues:**

   .. code-block:: bash

      # Recreate the virtual environment
      rm -rf .venv
      poetry env remove --all
      poetry install

2. **Pre-commit hook failures:**

   .. code-block:: bash

      # Update pre-commit hooks
      pre-commit autoupdate

      # Run hooks manually
      pre-commit run --all-files

3. **Documentation build errors:**

   .. code-block:: bash

      # Clean build directory
      cd docs
      make clean

      # Rebuild with verbose output
      sphinx-build -v --nitpicky --show-traceback --fail-on-warning --builder html docs/source docs/build/html

4. **Test failures:**

   .. code-block:: bash

      # Run tests with verbose output
      poetry run pytest -v ./clusx ./tests

      # Run a specific test
      poetry run pytest -v ./tests/test_specific_file.py::test_specific_function

5. **Cleaning build artifacts without make:**

   .. code-block:: bash

      # Remove Python cache files
      find ./ -name '__pycache__' -delete -o -name '*.pyc' -delete

      # Remove pytest cache
      rm -rf ./.pytest_cache

      # Remove coverage reports
      rm -rf ./coverage
