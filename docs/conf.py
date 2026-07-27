# -*- coding: utf-8 -*-
#
# Sphinx configuration for the cemd documentation.
# Full reference: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
 
project   = 'cemd'
copyright = '2022-2026, Jérôme Claverie'
author    = 'Jérôme Claverie'
version   = '0.1.0'
release   = '0.1.0'

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
 
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx_design',
    'numpydoc'
]
 
templates_path  = ['_templates']
source_suffix   = '.rst'
master_doc      = 'index'
language        = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style  = 'sphinx'
todo_include_todos = True
add_module_names = False

# ---------------------------------------------------------------------------
# Autodoc / Autosummary
# ---------------------------------------------------------------------------
 
autodoc_typehints    = 'none'
autodoc_member_order = 'bysource'
 
# Heavy optional dependencies — mock them so Sphinx does not need to import them
autodoc_mock_imports = [
    'pymatgen',
    'rdkit',
]

autodoc_default_options = {
    'show-inheritance': True,
    'undoc-members':    False,
    # Exclude instance attributes already documented in the class docstring
    # to avoid them appearing twice in the rendered page.
    'exclude-members': (
        'atoms, bonds, angles, dihedrals, impropers, velocities, '
        'pair_params, bond_params, angle_params, dihedral_params, improper_params'
    ),
}

autosummary_generate          = True
autosummary_imported_members  = True
 
# ---------------------------------------------------------------------------
# NumpyDoc
# ---------------------------------------------------------------------------
 
numpydoc_show_class_members      = False
numpydoc_class_members_toctree   = False
numpydoc_attributes_as_param_list = False

# ---------------------------------------------------------------------------
# Intersphinx — link to external docs
# ---------------------------------------------------------------------------
 
intersphinx_mapping = {
    'python':  ('https://docs.python.org/3',        None),
    'numpy':   ('https://numpy.org/doc/stable',     None),
    'pandas':  ('https://pandas.pydata.org/docs',   None),
    'MDAnalysis': ('https://docs.mdanalysis.org/stable',    None),
}


# ---------------------------------------------------------------------------
# HTML output — PyData Sphinx Theme
# ---------------------------------------------------------------------------
 
html_theme = 'pydata_sphinx_theme'
 
html_theme_options = {
    'logo': {
        'image_light': 'logo/cemd_logo.svg',
        'image_dark':  'logo/cemd_logo_dark.svg',
    },
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
}

html_show_sourcelink = False
 
html_static_path = ['_static']

html_css_files = [
    'https://fonts.googleapis.com/icon?family=Material+Icons',
    'custom.css',
]

html_sidebars = {
    "api/atomic_system": [],
    "installation": []

}

# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------
 
# latex_documents = [
#     (master_doc, 'cemd.tex', 'cemd Documentation', author, 'manual'),
# ]


# todo_include_todos = True

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'cemdsdoc'

# ---------------------------------------------------------------------------
# Other build (HTMLHelp, man, Texinfo, epub)
# ---------------------------------------------------------------------------
 
htmlhelp_basename = 'cemddoc'
 
man_pages = [
    (master_doc, 'cemd', 'cemd Documentation', [author], 1),
]
 
texinfo_documents = [
    (master_doc, 'cemd', 'cemd Documentation',
     author, 'cemd', 'Computational Elementary Matter Design.', 'Miscellaneous'),
]
 
epub_title           = project
epub_exclude_files   = ['search.html']

