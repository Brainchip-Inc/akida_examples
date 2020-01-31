# -- Project information -----------------------------------------------------

project = 'Akida Examples'
copyright = 'Copyright 2020, BrainChip Holdings Ltd. All Rights Reserved.'
author = 'Brainchip'

# The full version, including alpha/beta/rc tags
import pkg_resources

try:
    version = pkg_resources.get_distribution('akida').version
except Exception:
    version = 'unknown'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery',
    'autodocsumm'
]

# The suffix(es) of source filenames.
source_suffix = ['.rst']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

sphinx_gallery_conf = {
    'default_thumb_file': 'doc/img/brn.png',
    'examples_dirs': '../examples',
    'gallery_dirs': 'examples',
}

# Sphinx gallery will raise a Matplotlib agg warning that can only be silenced
# by filtering it:
import warnings

warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')

# -- Options for HTML output -------------------------------------------------
# Logo image displayed at the top of the sidebar
html_logo =  'img/akida.svg'

# Browser icon for tabs, windows and bookmarks
html_favicon = 'img/favicon.ico'

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': True,
    'collapse_navigation':False,
    'style_nav_header_background':'#3f51b5'
}

# Hide undesired components
html_show_sourcelink = False
html_show_sphinx = False

# -- Options for autodocsumm -------------------------------------------------
# autodocsumm allows to print a nice summary for API's auto generated doc
# see https://github.com/Chilipp/autodocsumm for more details
autodoc_default_options = {
    'autosummary': True,
}

# silence warning raised by autodocsumm
suppress_warnings = ['app.add_directive']
