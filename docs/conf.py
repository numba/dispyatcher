import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'dispyatcher'
copyright = '2023, Andre Masella'
author = 'Andre Masella'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon']
templates_path = ['_templates']
exclude_patterns = []
html_theme = 'alabaster'
html_static_path = ['_static']
