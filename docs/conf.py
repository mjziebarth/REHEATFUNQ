# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'REHEATFUNQ'
copyright = '2022-2024, Deutsches GeoForschungsZentrum Potsdam & Malte J. Ziebarth'
author = 'Malte J. Ziebarth'
release = '2.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinxcontrib.cairosvgconverter'
]

autodoc_typehints = "none"
add_module_names = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
html_theme_options = {
    "sidebar_width" : "7cm",
}

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'preamble' : r'''
    \usepackage{enumitem}
    \setlistdepth{10}
    '''
}
