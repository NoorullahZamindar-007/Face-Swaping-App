"""Route tests for the Face Swap app's upload flow.

The helpers tests stub Flask entirely, so the POST route and the 413
handler had zero coverage: the per-image upload limit, the magic-byte
check and the flash messages were only ever exercised by hand. These
tests use the real Flask test client with face_swap stubbed out.

dlib is not installed in the test environment and face_swap_utils
imports it at module level, so both are stubbed before importing the
app — the stubs only stand in for the swap itself, never for the
upload/validation code under test.
"""

import io
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# test_app_helpers.py installs minimal flask/werkzeug stubs in
# sys.modules when it runs first in the suite, and imports the app
# against them; the route tests need the real packages, so drop the
# stubs and any app module imported against them (a stub module has
# no __file__).
_dropped_stub = False
for _name in ("flask", "werkzeug", "werkzeug.utils", "face_swap_utils"):
    _mod = sys.modules.get(_name)
    if _mod is not None and getattr(_mod, "__file__", None) is None:
        del sys.modules[_name]
        _dropped_stub = True
if _dropped_stub:
    for _name in ("face_swap_app.app", "face_swap_app"):
        sys.modules.pop(_name, None)

if "dlib" not in sys.modules:
    sys.modules["dlib"] = types.ModuleType("dlib")

if "face_swap_utils" not in sys.modules:
    utils_stub = types.ModuleType("face_swap_utils")
    utils_stub.face_swap = lambda *a, **k: None
    sys.modules["face_swap_utils"] = utils_stub

import face_swap_app.app as app_module  # noqa: E402

# The root shim (imported by the helpers tests against the stub Flask)
# still points at the stub-bound app object; rebind it to the real app
# so test_app_copies_stay_in_sync keeps passing in a shared session.
_root_app = sys.modules.get("app")
if _root_app is not None:
    for _name, _value in vars(app_module).items():
        if not _name.startswith("__"):
            setattr(_root_app, _name, _value)

app = app_module.app


def _jpeg(size, seed=b"A"):
    """Valid-JPEG-magic bytes of exactly *size* bytes."""
    header = b"\xff\xd8\xff\xe0" + b"\x00\x10JFIF\x00"
    body = (seed * ((size // len(seed)) + 1))[: max(0, size - len(header))]
    return header + body


def _png(size):
    header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 4
    return header + b"B" * max(0, size - len(header))


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Test client writing uploads/results into tmp dirs, not the repo."""
    upload_dir = tmp_path / "uploads"
    result_dir = tmp_path / "result"
    upload_dir.mkdir()
    result_dir.mkdir()
    monkeypatch.setitem(app.config, "UPLOAD_FOLDER", str(upload_dir))
    monkeypatch.setitem(app.config, "RESULT_FOLDER", str(result_dir))
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def stub_swap(monkeypatch, tmp_path):
    """Replace face_swap with a stub that writes a dummy result file."""

    def fake_swap(path1, path2, result_path):
        Path(result_path).write_bytes(b"fake result")

    monkeypatch.setattr(app_module, "face_swap", fake_swap)


def _files(first, second):
    return {
        "image1": (io.BytesIO(first[1]), first[0]),
        "image2": (io.BytesIO(second[1]), second[0]),
    }


def test_get_index_shows_upload_limit(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"under 16 MB" in response.data


def test_missing_page_renders_error_page(client):
    # Any unknown URL used to show Flask's bare 404 page instead of
    # the app's own error page with a way back to the upload form.
    response = client.get("/no-such-page")

    assert response.status_code == 404
    assert b"404" in response.data
    assert b"Page not found" in response.data
    assert b"Back to the face swap form" in response.data


def test_post_missing_second_file_flashes(client):
    response = client.post(
        "/",
        data={"image1": (io.BytesIO(_jpeg(100)), "one.jpg")},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Please choose the second image." in response.data


def test_post_bad_extension_rejected(client):
    response = client.post(
        "/",
        data=_files(("one.txt", b"hello"), ("two.jpg", _jpeg(100))),
        follow_redirects=True,
    )

    assert b"Both uploads must be images" in response.data


def test_post_identical_files_rejected(client):
    data = _jpeg(100)
    response = client.post(
        "/",
        data=_files(("one.jpg", data), ("two.jpg", data)),
        follow_redirects=True,
    )

    assert b"Please upload two different images." in response.data


def test_post_renamed_text_file_rejected(client):
    response = client.post(
        "/",
        data=_files(("one.jpg", b"this is not an image"), ("two.jpg", _jpeg(100))),
        follow_redirects=True,
    )

    assert b"not a valid image" in response.data


def test_post_empty_files_report_invalid_image_not_same_upload(client):
    response = client.post(
        "/",
        data=_files(("one.jpg", b""), ("two.jpg", b"")),
        follow_redirects=True,
    )

    # Two empty files used to trip the same_upload check first and
    # wrongly report "two different images".
    assert b"not a valid image" in response.data
    assert b"two different images" not in response.data


def test_post_oversized_single_image_rejected(client, monkeypatch):
    monkeypatch.setattr(app_module, "_max_upload_mb", 1)

    response = client.post(
        "/",
        data=_files(("one.jpg", _jpeg(1024 * 1024 + 1)), ("two.jpg", _jpeg(100))),
        follow_redirects=True,
    )

    assert b"Each image must be under 1 MB." in response.data


def test_post_two_legal_images_combined_over_old_cap_succeeds(
    client, monkeypatch, stub_swap
):
    # The old MAX_CONTENT_LENGTH capped the whole request at one image's
    # limit, so two legal 0.6 MB images (1.2 MB combined) were rejected.
    monkeypatch.setattr(app_module, "_max_upload_mb", 1)
    monkeypatch.setitem(
        app.config, "MAX_CONTENT_LENGTH", (2 * 1024 * 1024) + 65536
    )

    response = client.post(
        "/",
        data=_files(
            ("one.jpg", _jpeg(600_000, seed=b"A")),
            ("two.jpg", _jpeg(600_000, seed=b"B")),
        ),
    )

    assert response.status_code == 200
    assert b"<h2>Result</h2>" in response.data


def test_post_failed_swap_flashes_error_and_leaves_no_orphans(
    client, monkeypatch
):
    # A failed swap (no face in one photo, a bad landmark file) must
    # report the error instead of a bare 500 — and must not leave the
    # uploaded face photos (or a half-written result) sitting in the
    # publicly served static folders for the prune to find later.
    def failing_swap(path1, path2, result_path):
        raise ValueError("No face detected in the first image.")

    monkeypatch.setattr(app_module, "face_swap", failing_swap)

    response = client.post(
        "/",
        data=_files(("one.jpg", _jpeg(100, seed=b"A")), ("two.jpg", _jpeg(100, seed=b"B"))),
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Face swap failed" in response.data
    assert b"No face detected in the first image." in response.data
    assert os.listdir(app.config["UPLOAD_FOLDER"]) == []
    assert os.listdir(app.config["RESULT_FOLDER"]) == []


def test_post_successful_swap_removes_uploaded_photos(
    client, monkeypatch, stub_swap
):
    # The swap output is already written when the route returns, so
    # the source face photos must not stay in the publicly served
    # static folder: only the result file may remain.
    response = client.post(
        "/",
        data=_files(("one.jpg", _jpeg(100, seed=b"A")), ("two.jpg", _jpeg(100, seed=b"B"))),
    )

    assert response.status_code == 200
    assert b"<h2>Result</h2>" in response.data
    assert os.listdir(app.config["UPLOAD_FOLDER"]) == []
    assert len(os.listdir(app.config["RESULT_FOLDER"])) == 1


def test_post_health_returns_branded_405_page(client):
    # /health only accepts GET: a POST used to show Flask's bare 405
    # page instead of the app's own error page with a way back to the form.
    response = client.post("/health")

    assert response.status_code == 405
    assert b"405" in response.data
    assert b"Method not allowed" in response.data
    assert b"Back to the face swap form" in response.data


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_server_error_renders_error_page(client, monkeypatch):
    # An unhandled exception used to show Flask's bare 500 page
    # instead of the app's own error page with a way back to the form.
    # TESTING must be False for Flask to invoke the 500 handler
    # instead of propagating the exception to the test.
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setitem(app.view_functions, "index", _boom)
    monkeypatch.setitem(app.config, "TESTING", False)

    response = client.get("/")

    assert response.status_code == 500
    assert b"500" in response.data
    assert b"Something went wrong" in response.data
    assert b"Back to the face swap form" in response.data


def test_post_over_request_limit_shows_413_message(client, monkeypatch):
    monkeypatch.setattr(app_module, "_max_upload_mb", 1)
    monkeypatch.setitem(app.config, "MAX_CONTENT_LENGTH", 100)

    response = client.post(
        "/",
        data=_files(("one.jpg", _jpeg(100)), ("two.jpg", _jpeg(100))),
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Image too large - please keep each upload under 1 MB." in response.data
