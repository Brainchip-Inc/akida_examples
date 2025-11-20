from datetime import datetime

# -- Project information -----------------------------------------------------

project = 'Akida Examples'
copyright = f'{datetime.now().year}, BrainChip Holdings Ltd. All Rights Reserved'
author = 'Brainchip'
version = 'Akida, 2nd Generation'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery', 'autodocsumm', 'sphinx.ext.viewcode',
    'sphinx_design', 'sphinxcontrib.video'
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
    'first_notebook_cell': ("%matplotlib inline"),
    'subsection_order':
        ExplicitOrder([
            '../examples/general', '../examples/quantization', '../examples/spatiotemporal',
            '../examples/edge'
        ]),
    'within_subsection_order':
        FileNameSortKey,
    'nested_sections': False
}

# Sphinx gallery will raise a Matplotlib agg warning that can only be silenced
# by filtering it:
import warnings

warnings.filterwarnings("ignore",
                        category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                        ' non-GUI backend, so cannot show the figure.')
# ONNXScript will raise a FutureWarning for "onnxscript.values.Op.param_schemas' is deprecated"
warnings.filterwarnings("ignore", category=FutureWarning, module="onnxscript")
# Optimum will raise warnings when using "main_export"
warnings.filterwarnings("ignore", module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch", message="Constant folding -*")

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
    'style_nav_header_background': '#000000'
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

# Silence warning raised by autodocsumm
suppress_warnings = ['app.add_directive', 'config.cache']

# This folder is copied to the documentation's HTML output
html_static_path = ['_static']

# Add a custom css file to remove rtd theme page width limit
html_css_files = [
    'custom.css',
]

# Add Leadlander tag for activity tracking
html_js_files = ['leadlander_tag.js']

# -- Exported variables -----------------------------------------------------
from importlib.metadata import version as importlib_version

akida_version = importlib_version('akida')
cnn2snn_version = importlib_version('cnn2snn')
models_version = importlib_version('akida-models')

from pip._internal.operations.freeze import freeze
pip_freeze = ', '.join([str(i) for i in freeze(local_only=True)])


def ultimateReplace(app, docname, source):
    result = source[0]
    for key in app.config.ultimate_replacements:
        result = result.replace(key, app.config.ultimate_replacements[key])
    source[0] = result


ultimate_replacements = {
    "{AKIDA_VERSION}": akida_version,
    "{CNN2SNN_VERSION}": cnn2snn_version,
    "{MODELS_VERSION}": models_version,
    "{PIP_FREEZE}": pip_freeze
}

# -- Link checks -----------------------------------------------------
# Ignore relative links and some specific links that prevent web scrapping (403 Client Error:
# Forbidden for url)
linkcheck_ignore = [
    "./.*", "../.*",
    "https://machinelearningmastery.com/object-recognition-with-deep-learning/",
    "https://www.sciencedirect.com/science/article/pii/S0893608018300108",
    "https://medium.com/.*"
]

# Ignore some anchors on github pages because checklink cannot resolve them
linkcheck_anchors_ignore = ["model", "confusion-matrix", "how-does-this-model-work"]

# Timeout for link checking in seconds
linkcheck_timeout = 20


def setup(app):
    app.add_config_value('ultimate_replacements', {}, True)
    app.connect('source-read', ultimateReplace)
