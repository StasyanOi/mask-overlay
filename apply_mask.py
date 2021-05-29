import csv

import cv2
import dlib
import numpy as np

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

cap = cv2.VideoCapture(0)  # webcame video

def getFacePoints(landmarks, points):
    dst_pts = []
    for i in range(len(points)):
        x = landmarks.part(points[i]).x
        y = landmarks.part(points[i]).y
        dst_pts.append(np.array([x, y]))
    return np.asarray(dst_pts).astype('float32')


def getMaskPoints(num):
    mask_annotation = "masks/" + str(num) + ".csv"
    with open(mask_annotation) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        src_pts = []
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                src_pts.append(np.array([float(row[1]), float(row[2])]))
            except ValueError:
                continue
    src_pts = np.array(src_pts, dtype="float32")
    return src_pts


mask_index = 1

while True:
    ret, img = cap.read()
    faces = face_detector(img, 1)
    landmark_tuple = []
    mask = None
    for k, d in enumerate(faces):
        landmarks = landmark_detector(img, d)
        dst_pts = getFacePoints(landmarks, [1, 3, 5, 7, 8, 9, 11, 13, 15, 29])
        src_pts = getMaskPoints(mask_index)
        if (dst_pts > 0).all():
            mask_img = cv2.imread("masks/" + str(mask_index) + ".png", cv2.IMREAD_UNCHANGED)
            mask_img = mask_img.astype(np.float32)
            mask_img = mask_img / 255
            M, _ = cv2.findHomography(src_pts, dst_pts)
            transformed_mask = cv2.warpPerspective(
                mask_img,
                M,
                (img.shape[1], img.shape[0]),
                None,
                cv2.INTER_LINEAR,
                cv2.BORDER_CONSTANT,
            )
            alpha_mask = transformed_mask[:, :, 3] * 255.0
            mask = alpha_mask
            alpha_image = 255.0 - alpha_mask
            for c in range(0, 3):
                img[:, :, c] = ((alpha_mask) * transformed_mask[:, :, c] + (alpha_image / 255) * img[:, :, c])

    cv2.imshow('img.jpg', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    if mask_index != 5:
        mask_index = mask_index + 1
    else:
        mask_index = 1

cap.release()
cv2.destroyAllWindows()
