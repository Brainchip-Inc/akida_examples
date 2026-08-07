from datetime import datetime

# -- Project information -----------------------------------------------------

project = 'MetaTF'
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
    'sphinx_design', 'sphinxcontrib.video', 'sphinx_reredirects'
]

# Redirect stubs keeping published deep links alive after the general examples
# were renumbered (PyTorch/ONNX workflow promoted to second position).
redirects = {
    "examples/general/plot_7_global_pytorch_workflow": "plot_1_global_pytorch_workflow.html",
    "examples/general/plot_1_akidanet_imagenet": "plot_2_akidanet_imagenet.html",
    "examples/general/plot_2_ds_cnn_kws": "plot_3_ds_cnn_kws.html",
    "examples/general/plot_3_regression": "plot_4_regression.html",
    "examples/general/plot_4_transfer_learning": "plot_5_transfer_learning.html",
    "examples/general/plot_5_voc_yolo_detection": "plot_6_voc_yolo_detection.html",
    "examples/general/plot_6_segmentation": "plot_7_segmentation.html",
}

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
    'style_nav_header_background': '#000000',
    "analytics_id": "G-T6Y7X9D33L",
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

# Root-level extra files, copied verbatim to the output root — holds the
# LLM-facing documentation index (https://llmstxt.org)
html_extra_path = ['_extra']

# Custom templates folder (used for GTM injection)
templates_path = ['_templates']

# Add a custom css file to remove rtd theme page width limit
html_css_files = [
    'custom.css',
]

# -- Exported variables -----------------------------------------------------
from importlib.metadata import version as importlib_version

akida_version = importlib_version('akida')
cnn2snn_version = importlib_version('cnn2snn')
models_version = importlib_version('akida-models')
metatf_version = importlib_version('metatf')

from pip._internal.operations.freeze import freeze
pip_freeze = ', '.join([str(i) for i in freeze(local_only=True)])

# -- Version switcher --------------------------------------------------------
# The RTD theme version flyout is only injected when hosted on readthedocs.org,
# so this self-hosted build renders _templates/versions.html instead, driven by
# the lists below. On release, add the archived version to legacy_doc_versions
# (it also feeds the "Previous versions" list on the Changelog page).
legacy_doc_versions = [
    '2.18.2', '2.17.0', '2.16.1', '2.15.0', '2.14.0', '2.13.0', '2.12.0',
    '2.11.0', '2.10.0', '2.9.0', '2.8.1', '2.7.2', '2.6.0', '2.4.0', '2.3.0'
]
legacy_doc_url = 'https://brainchip-inc.github.io/akida_examples_{}-doc-1/'

html_context = {
    'current_version': akida_version,
    'versions': [(akida_version, 'https://doc.brainchipinc.com/')] +
                [(v, legacy_doc_url.format(v)) for v in legacy_doc_versions],
}


def ultimateReplace(app, docname, source):
    result = source[0]
    for key in app.config.ultimate_replacements:
        result = result.replace(key, app.config.ultimate_replacements[key])
    source[0] = result


ultimate_replacements = {
    "{AKIDA_VERSION}": akida_version,
    "{CNN2SNN_VERSION}": cnn2snn_version,
    "{MODELS_VERSION}": models_version,
    "{METATF_VERSION}": metatf_version,
    "{PIP_FREEZE}": pip_freeze,
    "{DOC_VERSIONS}": '\n   '.join(
        f'* `{v}-doc-1 <{legacy_doc_url.format(v)}>`_' for v in legacy_doc_versions)
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
linkcheck_timeout = 60


def setup(app):
    app.add_config_value('ultimate_replacements', {}, True)
    app.connect('source-read', ultimateReplace)
    # ultimateReplace is a pure per-document string substitution, safe for
    # parallel builds (sphinx-build -j)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
