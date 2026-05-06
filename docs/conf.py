# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date

import eelbrain
from intersphinx_registry import get_intersphinx_mapping
from sphinx_gallery.sorting import FileNameSortKey

import ncrf

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NCRF'
author = 'Proloy Das and Christian Brodbeck'
copyright = f"{date.today().year}, {author}"  # noqa: A001
package = ncrf.__name__
gh_url = ""
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be extensions coming
# with Sphinx (named "sphinx.ext.*") or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.autosectionlabel",  # conflict with sphinx-gallery
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    'sphinx_gallery.gen_gallery',
    "sphinxcontrib.bibtex",
    "sphinx.ext.githubpages",  # .nojekyll file on generated HTML directory to publish the document on GitHub Pages. 
]

templates_path = ["templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Sphinx will warn about all references where the target cannot be found.
nitpicky = True
nitpick_ignore = [("py:obj", "optional"), ("py:obj", "NCRF")]

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = [f"{package}."]

# list of warning types to suppress
suppress_warnings = [
    "config.cache",
    # Sphinx-gallery creates duplicate labels:
    'autosectionlabel.sg_execution_times',
    'autosectionlabel.auto_examples/sg_execution_times',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_title = project

# -- autosummary -----------------------------------------------------------------------
autosummary_generate = True

# -- autodoc ---------------------------------------------------------------------------
autodoc_typehints = "none"
autodoc_member_order = "groupwise"
autodoc_warningiserror = True
autoclass_content = "class"

# -- napoleon --------------------------------------------------------------------------
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_keyword = True
napoleon_use_rtype = True

qualname_overrides = {
    "ncrf._model.NCRF": "ncrf.NCRF",
    "ncrf._model.RegressionData": "ncrf.RegressionData",
    "ncrf._ncrf.fit_ncrf": "ncrf.fit_ncrf",
}

# -- intersphinx -----------------------------------------------------------------------
intersphinx_mapping = get_intersphinx_mapping(
    packages={
        "matplotlib",
        "mne",
        "numpy",
        "pandas",
        "python",
        "scipy",
        "sklearn",
        "numba",
    }
)
intersphinx_mapping.update({
    "eelbrain": ("https://eelbrain.readthedocs.io/en/stable", None),
    'surfer': ('https://pysurfer.github.io', None),
})
intersphinx_timeout = 5

# -- sphinx-gallery

def use_pyplot(gallery_conf, fname):
    eelbrain.configure(frame=False)

sphinx_gallery_conf = {
    'examples_dirs': '../examples',   # path to your example scripts
    'filename_pattern': '/',
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'reset_modules': ('matplotlib', use_pyplot),
    "within_subsection_order": FileNameSortKey,
}


# -- sphinxcontrib-bibtex --------------------------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
suppress_warnings = ["bibtex.duplicate_citation", "config.cache"]
