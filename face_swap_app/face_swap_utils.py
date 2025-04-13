import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def extract_index_nparray(nparray):
    if nparray[0].size == 0:
        return None
    return nparray[0][0]

def face_swap(image1_path, image2_path, output_path):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(img1_gray)
    faces = detector(img1_gray)

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

        # For simplicity, just save the masked face for now
        cv2.imwrite(output_path, face_image_1)
