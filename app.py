"""Thin entry point for the Face Swap app.

The real implementation lives in ``face_swap_app/app.py``; this file used
to be a byte-identical duplicate kept in sync by hand, which drifted
whenever one copy was edited without the other. Import the package app
instead and re-export its namespace, so ``python app.py`` and
``python face_swap_app/app.py`` serve the same app object and
``import app`` keeps exposing the same helpers (``_env_int``,
``_valid_port``, ``allowed_file``, ...).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import face_swap_app.app as _real_app

for _name, _value in vars(_real_app).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

if __name__ == "__main__":
    # Debug mode only when explicitly enabled; never in production.
    # Host/port are overridable so container and PaaS hosts that assign $PORT
    # can start the app. A malformed PORT (e.g. "abc") falls back to 5000
    # instead of crashing startup, and out-of-range values (0, negative,
    # >65535) do the same: 0 would bind a random port the user cannot find.
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=_valid_port(_env_int("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
