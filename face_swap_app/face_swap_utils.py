import cv2
import numpy as np
import dlib
import os
import threading

MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"

_detector = None
_predictor = None
# The Flask dev server is threaded, so two concurrent first requests could
# both see unloaded models and each load the (large) dlib predictor.
_model_lock = threading.Lock()


def _load_models():
    """Load the dlib face detector and landmark predictor on first use.

    Raises a clear error instead of crashing at import time when the
    landmark model file is missing (it is not shipped with the repo).
    """
    global _detector, _predictor
    if _detector is None or _predictor is None:
        with _model_lock:
            if _detector is None or _predictor is None:
                candidates = [
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME),
                    os.path.abspath(MODEL_FILENAME),  # backward compatible: old CWD-relative path
                ]
                model_path = next((p for p in candidates if os.path.exists(p)), None)
                if model_path is None:
                    raise FileNotFoundError(
                        f"Required model file '{MODEL_FILENAME}' not found. Download it from "
                        "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "
                        "and place it next to face_swap_utils.py."
                    )
                _detector = dlib.get_frontal_face_detector()
                _predictor = dlib.shape_predictor(model_path)
    return _detector, _predictor


def _landmark_points(predictor, gray, face):
    """The 68 (x, y) landmark points for one detected face."""
    landmarks = predictor(gray, face)
    return [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]


def _triangle_indices(points):
    """Delaunay triangulation of *points*, as landmark-index triples.

    Subdiv2D returns triangle vertices as coordinates; each vertex is
    matched back to the landmark it came from, and a triangle is kept
    only when all three of its vertices are landmarks. Returns [] when
    the points cannot be triangulated (degenerate or duplicate points).
    """
    pts = np.array(points, np.int32)
    try:
        rect = cv2.boundingRect(cv2.convexHull(pts))
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert([tuple(p) for p in points])
        triangles = np.array(subdiv.getTriangleList(), dtype=np.int32)
    except cv2.error:
        return []
    indices = []
    for t in triangles:
        triple = []
        for k in (0, 2, 4):
            matches = np.where((pts == (t[k], t[k + 1])).all(axis=1))[0]
            if len(matches) == 0:
                triple = None
                break
            triple.append(int(matches[0]))
        if triple is not None:
            indices.append(triple)
    return indices


def _warp_face_onto(img1, points1, img2, points2, triangles):
    """Warp img1's face onto img2's face, one triangle at a time.

    The notebook's pipeline (Face_Swap.ipynb): each Delaunay triangle
    of the first face is cropped from img1, affine-warped onto the
    matching triangle of the second face, and painted onto a blank
    canvas the size of img2. Triangles whose affine transform is
    degenerate (collinear landmarks) are skipped instead of aborting
    the whole swap. Returns the canvas: the warped face on black.
    """
    new_face = np.zeros_like(img2)
    height, width = img2.shape[:2]
    for tri in triangles:
        src = np.array([points1[i] for i in tri], np.int32)
        dst = np.array([points2[i] for i in tri], np.int32)
        # Source triangle, cropped from the first image.
        x1, y1, w1, h1 = cv2.boundingRect(src)
        if x1 < 0 or y1 < 0 or x1 + w1 > img1.shape[1] or y1 + h1 > img1.shape[0]:
            continue
        cropped = img1[y1:y1 + h1, x1:x1 + w1]
        src_local = np.array([[p[0] - x1, p[1] - y1] for p in src], np.int32)
        # Destination triangle on the second face.
        x2, y2, w2, h2 = cv2.boundingRect(dst)
        if x2 < 0 or y2 < 0 or x2 + w2 > width or y2 + h2 > height:
            continue
        dst_local = np.array([[p[0] - x2, p[1] - y2] for p in dst], np.int32)
        dst_mask = np.zeros((h2, w2), np.uint8)
        cv2.fillConvexPoly(dst_mask, dst_local, 255)
        try:
            matrix = cv2.getAffineTransform(np.float32(src_local), np.float32(dst_local))
            warped = cv2.warpAffine(cropped, matrix, (w2, h2))
        except cv2.error:
            continue
        warped = cv2.bitwise_and(warped, warped, mask=dst_mask)
        # Reconstructing destination face: only paint where the canvas
        # is still empty, so overlapping triangles do not double up.
        area = new_face[y2:y2 + h2, x2:x2 + w2]
        area_gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        _, free = cv2.threshold(area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped = cv2.bitwise_and(warped, warped, mask=free)
        new_face[y2:y2 + h2, x2:x2 + w2] = cv2.add(area, warped)
    return new_face


def face_swap(image1_path, image2_path, output_path):
    """Put the first image's face onto the second image and save it.

    This is the pipeline from Face_Swap.ipynb: Delaunay-triangulate the
    first face's 68 landmarks, affine-warp each triangle onto the
    matching triangle of the second face, then seamless-clone the result
    onto the second photo. The previous implementation never moved any
    pixels between the images — it only masked the first image's face
    region onto a black background and called that a swap, while the
    second image was only checked for having a face.
    """
    detector, predictor = _load_models()

    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    if img1 is None:
        raise ValueError(f"Could not read image: {image1_path}")
    if img2 is None:
        raise ValueError(f"Could not read image: {image2_path}")

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    faces1 = detector(img1_gray)
    if len(faces1) == 0:
        raise ValueError("No face detected in the first image.")
    # The swap needs a face in both images: without this check a
    # faceless second image silently produced a masked first face
    # presented as a "swap" instead of a clear error.
    faces2 = detector(img2_gray)
    if len(faces2) == 0:
        raise ValueError("No face detected in the second image.")

    # The notebook triangulates one face per image: use the first face
    # found in each, so a group photo swaps the most prominent face
    # instead of looping over every face and overwriting the result.
    points1 = _landmark_points(predictor, img1_gray, faces1[0])
    points2 = _landmark_points(predictor, img2_gray, faces2[0])
    triangles = _triangle_indices(points1)
    if not triangles:
        raise ValueError("Could not triangulate the detected face.")
    new_face = _warp_face_onto(img1, points1, img2, points2, triangles)

    # Blend the warped face into the second photo: cut the second
    # face out, add the warped one in, then seamless-clone so the
    # edges match the second photo's lighting.
    hull2 = cv2.convexHull(np.array(points2, np.int32))
    head_mask = np.zeros_like(img2_gray)
    cv2.fillConvexPoly(head_mask, hull2, 255)
    no_face = cv2.bitwise_and(img2, img2, mask=cv2.bitwise_not(head_mask))
    combined = cv2.add(no_face, new_face)
    x, y, w, h = cv2.boundingRect(hull2)
    center = (x + w // 2, y + h // 2)
    try:
        swapped = cv2.seamlessClone(combined, img2, head_mask, center, cv2.NORMAL_CLONE)
    except cv2.error as exc:
        raise RuntimeError("Could not blend the swapped face into the second image.") from exc

    # Write once, after the swap is complete: writing inside the loop
    # rewrote the file once per face (and left a half-masked image on
    # disk if a later face failed), while the final file was always
    # the cumulative mask computed here.
    # imwrite returns False (it does not raise) when the disk is full
    # or the path is unwritable, so verify it before claiming success.
    if not cv2.imwrite(output_path, swapped):
        raise IOError(f"Could not write the swapped image to: {output_path}")
