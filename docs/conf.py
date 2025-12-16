import os
import sys
# o la ruta on està portcanto_esquelet
sys.path.insert(0, os.path.abspath('..'))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # si vols suport per a docstrings tipus Google/NumPy
]
