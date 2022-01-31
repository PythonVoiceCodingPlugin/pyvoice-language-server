"""Sphinx configuration."""
from datetime import datetime

project = "pyvoice"
author = "mpourmpoulis"
copyright = f"{datetime.now().year}, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_click",
    "sphinx_rtd_theme",
]
autodoc_typehints = "description"
html_theme = "sphinx_rtd_theme"
