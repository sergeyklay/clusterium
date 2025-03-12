Contributing
============

If you would like to contribute to Clusterium, please take a look at the
`current issues <https://github.com/sergeyklay/clusterium/issues>`_.  If there is
a bug or feature that you want but it isn't listed, make an issue and work on it.

Bug reports
-----------

*Before raising an issue, please ensure that you are using the latest version
of Clusterium.*

Please provide the following information with your issue to enable us to
respond as quickly as possible.

* The relevant versions of the packages you are using.
* The steps to recreate your issue.
* The full stacktrace if there is an exception.
* An executable code example where possible

Guidelines for bug reports:

* **Use the GitHub issue search** — check if the issue has already been
  reported.
* **Check if the issue has been fixed** — try to reproduce it using the latest
  ``main`` branch in the repository.
* Isolate the problem — create a reduced test case and a live example.

A good bug report shouldn't leave others needing to chase you up for more
information. Please try to be as detailed as possible in your report. What is
your environment? What steps will reproduce the issue? What OS experiences the
problem? What would you expect to be the outcome? All these details will help
people to fix any potential bugs.

Feature requests
----------------

Feature requests are welcome. But take a moment to find out whether your idea
fits with the scope and aims of the project. It's up to *you* to make a strong
case to convince the project's developers of the merits of this feature. Please
provide as much detail and context as possible.

Pull requests
-------------

Good pull requests - patches, improvements, new features - are a fantastic
help. They should remain focused in scope and avoid containing unrelated
commits.

Follow this process if you'd like your work considered for inclusion in the
project:

1. Check for open issues or open a fresh issue to start a discussion around a
   feature idea or a bug.
2. Fork `the repository <https://github.com/sergeyklay/clusterium>`_
   on GitHub to start making your changes to the ``main`` branch
   (or branch off of it).
3. Write a test which shows that the bug was fixed or that the feature works as
   expected.
4. Send a pull request and bug the maintainer until it gets merged and published.

If you are intending to implement a fairly large feature we'd appreciate if you
open an issue with GitHub detailing your use case and intended solution to
discuss how it might impact other work that is in flight.

Development setup
-----------------

Below is a quick setup guide. For a more comprehensive guide, please refer to the
`Maintainers Guide <https://clusterium.readthedocs.io/en/latest/maintainers.html>`_.

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/sergeyklay/clusterium.git
      cd clusterium

2. Install dependencies using Poetry:

   .. code-block:: bash

      poetry install

3. Set up pre-commit hooks:

   .. code-block:: bash

      poetry run pre-commit install

4. Run tests to ensure everything is working:

   .. code-block:: bash

      poetry run pytest

Code style
----------

This project uses:

* `Black <https://black.readthedocs.io/>`_ for code formatting
* `isort <https://pycqa.github.io/isort/>`_ for import sorting
* `flake8 <https://flake8.pycqa.org/>`_ for linting

These tools are automatically run when you use pre-commit hooks.

Testing
-------

We use pytest for testing. Please ensure that your code includes appropriate tests.
To run tests:

.. code-block:: bash

   poetry run pytest

To run tests with coverage:

.. code-block:: bash

   poetry run coverage run -m pytest
   poetry run coverage report

**By submitting a patch, you agree to allow the project owner to license your
work under the same license as that used by the project.**

Resources
---------

* `How to Contribute to Open Source <https://opensource.guide/how-to-contribute/>`_
* `Using Pull Requests <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_
* `Writing good commit messages <https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html>`_
