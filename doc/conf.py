# -- Project information -----------------------------------------------------

project = 'Akida Examples'
copyright = '2022, BrainChip Holdings Ltd. All Rights Reserved'
author = 'Brainchip'
version = 'Akida, 2nd Generation'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery', 'autodocsumm', 'sphinx.ext.viewcode'
]

# The suffix(es) of source filenames.
source_suffix = ['.rst']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

sphinx_gallery_conf = {
    'default_thumb_file':
        'doc/img/brn.png',
    'examples_dirs':
        '../examples',
    'gallery_dirs':
        'examples',
    'remove_config_comments':
        True,
    'first_notebook_cell': ("%matplotlib notebook"),
    'subsection_order':
        ExplicitOrder([
            '../examples/general', '../examples/cnn2snn', '../examples/edge'
        ]),
    'within_subsection_order':
        FileNameSortKey
}

# Sphinx gallery will raise a Matplotlib agg warning that can only be silenced
# by filtering it:
import warnings

warnings.filterwarnings("ignore",
                        category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                        ' non-GUI backend, so cannot show the figure.')

# -- Options for HTML output -------------------------------------------------
# Logo image displayed at the top of the sidebar
html_logo = 'img/MetaTF_logo.png'

# Browser icon for tabs, windows and bookmarks
html_favicon = 'img/favicon.ico'

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': True,
    'collapse_navigation': False,
    'style_nav_header_background': '#989898'
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

# this folder is copied to the documentation's HTML output
html_static_path = ['_static']

# add a custom css file to remove rtd theme page width limit
html_css_files = [
    'custom.css',
]

# -- Exported variables -----------------------------------------------------
import pkg_resources

akida_version = pkg_resources.get_distribution('akida').version
cnn2snn_version = pkg_resources.get_distribution('cnn2snn').version
models_version = pkg_resources.get_distribution('akida-models').version


def ultimateReplace(app, docname, source):
    result = source[0]
    for key in app.config.ultimate_replacements:
        result = result.replace(key, app.config.ultimate_replacements[key])
    source[0] = result


ultimate_replacements = {
    "{AKIDA_VERSION}": akida_version,
    "{CNN2SNN_VERSION}": cnn2snn_version,
    "{MODELS_VERSION}": models_version
}


def setup(app):
    app.add_config_value('ultimate_replacements', {}, True)
    app.connect('source-read', ultimateReplace)
