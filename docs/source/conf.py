import os
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../src'))


import datetime
import numgraph
import doctest

project = 'NumGraph'
author = 'Alessio Gravina and Danilo Numeroso'
copyright = f'{datetime.datetime.now().year}, {author}' 
release = numgraph.__version__
version = numgraph.__version__

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = 'bysource'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autodoc.typehints',
              'sphinx.ext.autosummary',
              'sphinx.ext.coverage', 
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'sphinx.ext.viewcode'            
]

autosummary_generate = True
autodoc_typehints = "none" #'description'

napoleon_type_aliases = {
    'NDArray': 'numpy.typing.NDArray',
    'Generator': 'numpy.random.Generator',
    'Optional': 'typing.Optional',
    #'Tuple': 'typing.Tuple',
    'Callable': 'typing.Callable'
}

#napoleon_preprocess_types = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_rtype = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_logo = f'{html_static_path[0]}/img/NumGraph_logo.svg'
html_favicon = f'{html_static_path[0]}/img/NumGraph_favicon.svg'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
    'collapse_navigation': False,
    'style_nav_header_background': '#EFEFEF',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None)
}
