# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

sys.path.insert(0, str(Path("..", "..", "clusx").resolve()))

from clusx import __author__, __copyright__, __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Clusterium"
author = __author__
copyright = __copyright__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for nitpick -----------------------------------------------------

# In nitpick mode (-n), still ignore any of the following "broken" references
# to non-types.
nitpick_ignore = [
    ("py:class", "np.float32"),
    ("py:class", "np.int64"),
    ("py:class", "np.ndarray"),
    ("py:class", "numpy.bool_"),
    ("py:class", "NDArray"),
    ("py:class", "Axes"),
    ("py:class", "SentenceTransformer"),
]

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# Output file base name for HTML help builder.
htmlhelp_basename = "clusterium-doc"
