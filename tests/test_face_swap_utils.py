"""Regression tests for face_swap_app/face_swap_utils.py.

dlib is not installed in the test environment, so it is stubbed and
_load_models is monkeypatched to return a fake detector/predictor;
cv2 is real, so the triangulation, warp and seamless-clone code runs
for real. Runnable with pytest or directly:
``python3 tests/test_face_swap_utils.py``.
"""

import math
import sys
import types
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# face_swap_utils imports dlib at module level; stub it before the real
# module loads. test_app_helpers.py also stubs the utils module itself
# in sys.modules, so drop that stub first.
if "dlib" not in sys.modules:
    sys.modules["dlib"] = types.ModuleType("dlib")
sys.modules.pop("face_swap_utils", None)
sys.modules.pop("face_swap_app.face_swap_utils", None)

import face_swap_app.face_swap_utils as utils  # noqa: E402


class _FakePoint:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _FakeShape:
    def __init__(self, points):
        self._points = points

    def part(self, n):
        x, y = self._points[n]
        return _FakePoint(x, y)


class _FakePredictor:
    """Returns the same 68 landmarks for any face."""

    def __init__(self, points):
        self._points = points

    def __call__(self, gray, face):
        return _FakeShape(self._points)


class _FakeDetector:
    """Returns one queued result per call, so the two images can differ."""

    def __init__(self, results):
        self._results = list(results)

    def __call__(self, gray):
        return self._results.pop(0) if self._results else []


def _face_points(cx=100, cy=100):
    """68 distinct landmarks spread over a disc, like a real face."""
    points = []
    for n in range(68):
        angle = 2 * math.pi * n / 68
        radius = 25 + 15 * (n % 3)
        points.append(
            (int(cx + radius * math.cos(angle)), int(cy + radius * math.sin(angle)))
        )
    return points


def _write_solid_image(path, color_bgr, size=200):
    img = np.full((size, size, 3), color_bgr, np.uint8)
    assert cv2.imwrite(str(path), img)


def _fake_models(monkeypatch, points=None, detector_results=None):
    points = points if points is not None else _face_points()
    detector = _FakeDetector(
        detector_results if detector_results is not None else [[object()], [object()]]
    )
    monkeypatch.setattr(utils, "_load_models", lambda: (detector, _FakePredictor(points)))


def test_triangle_indices_cover_the_face():
    triangles = utils._triangle_indices(_face_points())
    # A disc of 68 points triangulates into dozens of triangles, all
    # referencing real landmarks.
    assert len(triangles) > 10
    for triple in triangles:
        assert len(triple) == 3
        assert all(0 <= i < 68 for i in triple)
        assert len(set(triple)) == 3


def test_triangle_indices_reject_degenerate_points():
    # All 68 landmarks on one spot cannot be triangulated.
    assert utils._triangle_indices([(100, 100)] * 68) == []


def _write_textured_face_image(path):
    """First test photo: a blue scene with a textured red face.

    The face is a red disc with a dark-red grid, not a flat colour:
    seamlessClone's Poisson blend erases flat colours (zero gradient
    means the interior just takes the destination colour), so a flat
    red face would vanish from the result even when the swap works.
    Texture keeps the first face's red visible after blending.
    """
    img = np.full((200, 200, 3), (255, 0, 0), np.uint8)
    cv2.circle(img, (100, 100), 55, (0, 0, 255), -1)
    for x in range(50, 151, 15):
        cv2.line(img, (x, 45), (x, 155), (0, 0, 128), 3)
    for y in range(50, 151, 15):
        cv2.line(img, (45, y), (155, y), (0, 0, 128), 3)
    assert cv2.imwrite(str(path), img)


def test_swap_puts_first_faces_pixels_onto_second_photo(tmp_path, monkeypatch):
    # The first image carries a textured red face, the second is a
    # solid blue photo (BGR, PNG so no compression noise). A real swap
    # puts the first face onto the second photo: the face region must
    # pick up the first image's red while the background stays the
    # second photo's blue. The old implementation only masked the
    # first image, so its output was red everywhere with a black
    # background — this test fails on it.
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    output = tmp_path / "swapped.png"
    _write_textured_face_image(first)
    _write_solid_image(second, (255, 0, 0))
    _fake_models(monkeypatch)

    utils.face_swap(str(first), str(second), str(output))

    result = cv2.imread(str(output))
    assert result is not None
    assert result.shape == (200, 200, 3)
    # Inside the face: the first image's red survives the blend.
    for y, x in ((100, 100), (120, 120)):
        blue, green, red = (int(v) for v in result[y, x])
        assert red > 40, (y, x, blue, green, red)
    # Corner, outside the face hull: still the second photo's blue.
    blue, green, red = (int(v) for v in result[5, 5])
    assert blue > 200 and red < 20, (blue, green, red)


def test_swap_rejects_unreadable_images(tmp_path, monkeypatch):
    second = tmp_path / "second.jpg"
    _write_solid_image(second, (255, 0, 0))
    _fake_models(monkeypatch)

    with pytest.raises(ValueError, match="Could not read image"):
        utils.face_swap(str(tmp_path / "missing.jpg"), str(second), str(tmp_path / "out.jpg"))
    with pytest.raises(ValueError, match="Could not read image"):
        utils.face_swap(str(second), str(tmp_path / "missing.jpg"), str(tmp_path / "out.jpg"))


def test_swap_rejects_a_missing_face_in_either_image(tmp_path, monkeypatch):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _write_solid_image(first, (0, 0, 255))
    _write_solid_image(second, (255, 0, 0))

    _fake_models(monkeypatch, detector_results=[[], [object()]])
    with pytest.raises(ValueError, match="No face detected in the first image"):
        utils.face_swap(str(first), str(second), str(tmp_path / "out.jpg"))

    _fake_models(monkeypatch, detector_results=[[object()], []])
    with pytest.raises(ValueError, match="No face detected in the second image"):
        utils.face_swap(str(first), str(second), str(tmp_path / "out.jpg"))


def test_swap_rejects_degenerate_landmarks(tmp_path, monkeypatch):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _write_solid_image(first, (0, 0, 255))
    _write_solid_image(second, (255, 0, 0))
    _fake_models(monkeypatch, points=[(100, 100)] * 68)

    with pytest.raises(ValueError, match="Could not triangulate"):
        utils.face_swap(str(first), str(second), str(tmp_path / "out.jpg"))


def test_swap_reports_an_unwritable_output(tmp_path, monkeypatch):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _write_solid_image(first, (0, 0, 255))
    _write_solid_image(second, (255, 0, 0))
    _fake_models(monkeypatch)
    # imwrite returns False (it does not raise) when the output cannot
    # be written; the swap must report it instead of claiming success.
    monkeypatch.setattr(cv2, "imwrite", lambda *args, **kwargs: False)

    with pytest.raises(IOError, match="Could not write"):
        utils.face_swap(str(first), str(second), str(tmp_path / "out.jpg"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
