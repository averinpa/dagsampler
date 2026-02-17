import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../src"))

project = "dagsampler"
author = "Pavel Averin"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

html_theme = "furo"
html_static_path = ["_static"]
