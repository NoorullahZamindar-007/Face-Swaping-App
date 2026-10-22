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

def extract_index_nparray(nparray):
    if nparray[0].size == 0:
        return None
    return nparray[0][0]

def face_swap(image1_path, image2_path, output_path):
    detector, predictor = _load_models()

    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    if img1 is None:
        raise ValueError(f"Could not read image: {image1_path}")
    if img2 is None:
        raise ValueError(f"Could not read image: {image2_path}")

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(img1_gray)
    faces = detector(img1_gray)
    if len(faces) == 0:
        raise ValueError("No face detected in the first image.")

    for face in faces:
        landmarks = predictor(img1_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))

        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, convexhull, 255)

        face_image_1 = cv2.bitwise_and(img1, img1, mask=mask)

        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)

        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            index_pt1 = extract_index_nparray(np.where((points == pt1).all(axis=1)))
            index_pt2 = extract_index_nparray(np.where((points == pt2).all(axis=1)))
            index_pt3 = extract_index_nparray(np.where((points == pt3).all(axis=1)))

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                indexes_triangles.append([index_pt1, index_pt2, index_pt3])

        # For simplicity, just save the masked face for now.
        # imwrite returns False (it does not raise) when the disk is full
        # or the path is unwritable, so verify it before claiming success.
        if not cv2.imwrite(output_path, face_image_1):
            raise IOError(f"Could not write the swapped image to: {output_path}")
