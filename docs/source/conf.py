# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
import satellighte as sat


# -- Project information -----------------------------------------------------

project = "Satellighte"
copyright = sat.__copyright__
author = sat.__author__

# The full version, including alpha/beta/rc tags
release = sat.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    "sphinx.ext.intersphinx",  # Link to other project's documentation (see mapping below)
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
    "nbsphinx",  # Integrate Jupyter Notebooks and Sphinx
    "sphinxemoji.sphinxemoji",
    "sphinxcontrib.napoleon",
    "recommonmark",
    "sphinx_markdown_tables",
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = (
    False  # Remove 'view source code' from top of page (for html, not python)
)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
nbsphinx_allow_errors = True  # Continue through Jupyter errors
# autodoc_typehints = "description" # Sphinx-native method. Not as good as sphinx_autodoc_typehints
add_module_names = False  # Remove namespaces from class/method signatures

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_parsers = {
    ".md": "recommonmark.parser.CommonMarkParser",
}

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
# html_theme = "renku"
# html_permalinks_icon = "<span>#</span>"
# html_theme = "sphinxawesome_theme"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
