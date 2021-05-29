import cv2
import dlib
import os
import csv
import numpy as np
import math

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
specs_ori = cv2.imread('masks/balaclavas/balaclava_rgba.png', -1)

cap = cv2.VideoCapture(0)  # webcame video


def getPoints(landmarks, points):
    dst_pts = []
    for i in range(len(points)):
        x = landmarks.part(points[i]).x
        y = landmarks.part(points[i]).y
        dst_pts.append(np.array([x, y]))
    return np.asarray(dst_pts).astype('float32')


def getSrcPoints(num):
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


def isFrontal(landmarks, x_delta, y_delta):
    horizontal_center = landmarks.part(28).x
    horizontal_right = landmarks.part(16).x
    horizontal_left = landmarks.part(0).x

    vertical_low = landmarks.part(30).x
    vertical_high = landmarks.part(27).x

    diff_y = math.fabs((horizontal_right - horizontal_center) - horizontal_center + horizontal_left)
    diff_x = math.fabs(vertical_low - vertical_high)

    if (diff_x <= x_delta) & (diff_y <= y_delta):
        return True
    else:
        return False
    pass


mask_index = 1


def sort_names(dir):
    ints = []
    for i in range(len(dir)):
        ints.append(int(dir[i].split(".")[0]))
    ints.sort()
    for i in range(len(dir)):
        dir[i] = str(ints[i]) + ".png"
    return dir


dir = os.listdir("CelebAMask-HQ/CelebA-HQ-img-256-256/")
dir = sort_names(dir)
for i in range(len(dir) - 20000):
    print(i)
    img = cv2.imread("CelebAMask-HQ/CelebA-HQ-img-256-256/" + dir[i], cv2.IMREAD_UNCHANGED)
    faces = face_detector(img, 1)
    landmark_tuple = []
    mask = None
    for k, d in enumerate(faces):
        landmarks = landmark_detector(img, d)
        dst_pts = getPoints(landmarks, [1, 3, 5, 7, 8, 9, 11, 13, 15, 29])
        src_pts = getSrcPoints(mask_index)
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

    if mask is not None:
        cv2.imwrite("CelebAMask-HQ/CelebA-HQ-img-256-256-labels/" + dir[i], mask * 255)
        cv2.imwrite("CelebAMask-HQ/CelebA-HQ-img-256-256-masked/" + dir[i], img)

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
