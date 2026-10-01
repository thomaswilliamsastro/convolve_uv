import datetime
import os
import sys
import tomllib
from importlib.metadata import version as package_version
from pathlib import Path

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Read info from the pyproject.toml file

package_dir = Path(__file__).parent.parent
pyproject_path = os.path.join(package_dir, "pyproject.toml")

with open(pyproject_path, "rb") as metadata_file:
    metadata = tomllib.load(metadata_file)["project"]

project = metadata["name"]
author = metadata["authors"][0]["name"]
copyright = f"{datetime.datetime.now(tz=datetime.UTC).year}, {author}"

# The version of the installed package, which setuptools_scm derives from the latest
# ``v*`` git tag. ``release`` is the full version (e.g. "0.3.1.dev24+g5021eba2e"), and
# ``version`` the short X.Y form that Sphinx expects (e.g. "0.3").
release = package_version(project)
version = ".".join(release.split(".")[:2])

sys.path.append(str(package_dir))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_automodapi.automodapi",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
]

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "globaltoc_collapse": False,
    "logo_only": True,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

autoclass_content = "both"
todo_include_todos = True
coverage_show_missing_items = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

master_doc = "index"
html_logo = "images/convolve_uv.png"
html_theme = "sphinx_rtd_theme"
