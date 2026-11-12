"""Thin re-export of the Face Swap utils.

The real implementation lives in ``face_swap_app/face_swap_utils.py``; this file used
to be a byte-identical duplicate kept in sync by hand, which drifted whenever one
copy was edited without the other. Import the package module instead and re-export
its namespace, so ``python app.py`` and ``python face_swap_app/app.py`` use the same
``face_swap`` — which copy got imported used to depend on the launch directory.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import face_swap_app.face_swap_utils as _real_utils

for _name, _value in vars(_real_utils).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
