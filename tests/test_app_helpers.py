"""Regression tests for the Face Swap app's pure helpers.

app.py imports Flask, cv2 and dlib at module level, none of which are
guaranteed in a test environment, so minimal stubs stand in for them;
the tests only exercise the pure helpers, never the camera/model code.
Runnable with pytest or directly: ``python3 tests/test_app_helpers.py``.
"""

import io
import os
import sys
import time
import types
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "flask" not in sys.modules:
    flask_stub = types.ModuleType("flask")

    class _Flask:
        def __init__(self, *args, **kwargs):
            self.config = {}
            self.template_folder = kwargs.get("template_folder")
            self.secret_key = None

        def route(self, *args, **kwargs):
            def deco(func):
                return func

            return deco

        def errorhandler(self, *args, **kwargs):
            def deco(func):
                return func

            return deco

    flask_stub.Flask = _Flask
    flask_stub.render_template = lambda *a, **k: ""
    flask_stub.request = types.SimpleNamespace(form={}, files={})
    flask_stub.redirect = lambda *a, **k: None
    flask_stub.url_for = lambda *a, **k: ""
    flask_stub.flash = lambda *a, **k: None
    sys.modules["flask"] = flask_stub

if "face_swap_utils" not in sys.modules:
    utils_stub = types.ModuleType("face_swap_utils")
    utils_stub.face_swap = lambda *a, **k: None
    sys.modules["face_swap_utils"] = utils_stub

if "werkzeug.utils" not in sys.modules:
    werkzeug_stub = types.ModuleType("werkzeug")
    utils_mod = types.ModuleType("werkzeug.utils")
    utils_mod.secure_filename = lambda name: name
    werkzeug_stub.utils = utils_mod
    sys.modules["werkzeug"] = werkzeug_stub
    sys.modules["werkzeug.utils"] = utils_mod

# Importing app creates static/uploads and static/result next to itself;
# keep the checkout clean by no-op'ing makedirs for the import.
with mock.patch.object(os, "makedirs"):
    import app  # noqa: E402  (import after the stubs is intentional)


def test_allowed_file_is_case_insensitive_and_rejects_double_extensions():
    assert app.allowed_file("photo.JPG")
    assert app.allowed_file("photo.png")
    assert app.allowed_file("photo.webp")
    assert not app.allowed_file("photo")
    assert not app.allowed_file("photo.jpg.exe")
    assert not app.allowed_file("archive.tar.gz")


def test_unique_filename_prefixes_a_random_id():
    first = app.unique_filename("photo.jpg")
    second = app.unique_filename("photo.jpg")
    assert first != second
    assert first.endswith("_photo.jpg")
    assert second.endswith("_photo.jpg")
    assert len(first.split("_")[0]) == 8


def test_env_int_falls_back_on_bad_values(monkeypatch):
    monkeypatch.setenv("FACE_SWAP_TEST_INT", "not-a-number")
    assert app._env_int("FACE_SWAP_TEST_INT", 16) == 16
    monkeypatch.setenv("FACE_SWAP_TEST_INT", "42")
    assert app._env_int("FACE_SWAP_TEST_INT", 16) == 42
    monkeypatch.delenv("FACE_SWAP_TEST_INT", raising=False)
    assert app._env_int("FACE_SWAP_TEST_INT", 16) == 16


def test_prune_stale_files_removes_only_old_files(tmp_path):
    old = tmp_path / "old.jpg"
    fresh = tmp_path / "fresh.jpg"
    old.write_bytes(b"x")
    fresh.write_bytes(b"x")
    old_time = time.time() - 1000
    os.utime(old, (old_time, old_time))

    assert app._prune_stale_files(str(tmp_path), 100) == 1
    assert not old.exists()
    assert fresh.exists()


def test_prune_stale_files_negative_window_removes_nothing(tmp_path):
    stale = tmp_path / "stale.jpg"
    stale.write_bytes(b"x")
    old_time = time.time() - 100000
    os.utime(stale, (old_time, old_time))

    # A negative retention window is invalid config, not "delete
    # everything": without the guard this would wipe the folder.
    assert app._prune_stale_files(str(tmp_path), -1) == 0
    assert stale.exists()


def test_user_facing_error_strips_server_paths():
    message = app._user_facing_error(
        RuntimeError("Could not read image: /srv/app/static/uploads/abc_photo.jpg"),
        "/srv/app/static/uploads/abc_photo.jpg",
    )
    assert "/srv/app" not in message
    assert "abc_photo.jpg" in message


def test_valid_port_rejects_out_of_range_values():
    # A PORT typo used to crash app.run at startup instead of starting
    # on the default port; 0 would bind a random port the user cannot find.
    assert app._valid_port(5000) == 5000
    assert app._valid_port(8080) == 8080
    assert app._valid_port(1) == 1
    assert app._valid_port(65535) == 65535
    assert app._valid_port(0) == 5000
    assert app._valid_port(-1) == 5000
    assert app._valid_port(65536) == 5000
    assert app._valid_port(70000) == 5000
    assert app._valid_port("abc") == 5000
    assert app._valid_port(None) == 5000


def test_same_upload_detects_identical_files_and_rewinds():
    # Uploading the same photo twice used to run the swap and render a
    # masked copy of the first image as a "result". The streams must be
    # rewound afterwards, or the later .save() calls would write empty
    # files.
    first = io.BytesIO(b"fake-image-bytes")
    second = io.BytesIO(b"fake-image-bytes")
    assert app.same_upload(first, second) is True
    assert first.read() == b"fake-image-bytes"
    assert second.read() == b"fake-image-bytes"

    assert app.same_upload(io.BytesIO(b"aaa"), io.BytesIO(b"bbb")) is False
    assert app.same_upload(io.BytesIO(b"aaa"), io.BytesIO(b"aaaa")) is False


class _UnreadableUpload(io.BytesIO):
    def read(self, *args, **kwargs):
        raise OSError("disk error")


def test_same_upload_returns_false_when_a_stream_cannot_be_read():
    # A read failure must never block a legitimate upload.
    assert app.same_upload(_UnreadableUpload(b"aaa"), io.BytesIO(b"aaa")) is False
    assert app.same_upload(io.BytesIO(b"aaa"), _UnreadableUpload(b"aaa")) is False


def test_prune_stale_files_missing_folder_removes_nothing(tmp_path):
    # A folder that does not exist (fresh clone, never uploaded) must
    # not raise inside the per-request prune.
    assert app._prune_stale_files(str(tmp_path / "no-such-folder"), 100) == 0


def test_app_copies_stay_in_sync():
    # Root app.py is now a thin shim, not a second copy: it must serve
    # the very same Flask app object as face_swap_app/app.py.
    import face_swap_app.app as real_app

    assert app.app is real_app.app


def test_utils_shim_resolves_to_the_same_functions(monkeypatch):
    # Root face_swap_utils.py used to be a hand-kept mirror of
    # face_swap_app/face_swap_utils.py, guarded only by a byte-identity
    # test. It is now a thin shim: it must expose the very same
    # functions as the package module.
    # The test module stubs face_swap_utils (for the app import above)
    # and cv2/dlib are not installed in the test env, so stub them and
    # load the real modules fresh for this check.
    for name in ("cv2", "dlib"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.delitem(sys.modules, "face_swap_utils", raising=False)
    monkeypatch.delitem(sys.modules, "face_swap_app.face_swap_utils", raising=False)

    import face_swap_app.face_swap_utils as real_utils
    import face_swap_utils as root_utils

    assert root_utils.face_swap is real_utils.face_swap
    assert root_utils._load_models is real_utils._load_models
    assert root_utils.MODEL_FILENAME == real_utils.MODEL_FILENAME


def test_template_folder_points_at_real_templates():
    # The root copy used to rely on Flask's default templates/ folder,
    # which does not exist at the repo root, so every GET / 500'd.
    template_dir = Path(app.app.template_folder)
    assert (template_dir / "index.html").exists()


if __name__ == "__main__":
    for _name, _func in sorted(globals().items()):
        if _name.startswith("test_") and callable(_func):
            import inspect

            if "tmp_path" in inspect.signature(_func).parameters or "monkeypatch" in inspect.signature(
                _func
            ).parameters:
                continue  # fixture-based tests run under pytest only
            _func()
    print("all face-swap helper tests pass")
