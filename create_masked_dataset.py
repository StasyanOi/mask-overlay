import cv2
import dlib
import os
import csv
import numpy as np
import math
import PIL.Image as Image
from tensorflow.keras.models import load_model

# dir = "train_data/medical/CelebA-HQ-img-256-256-masked"

# model = load_model("unet_seg", compile=False)
# model.summary()
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

# faces = face_detector(img, 1)
#
# landmark_tuple = []
# for k, d in enumerate(faces):
#     landmarks = landmark_detector(img, d)
#     for n in range(0, 81):
#         x = landmarks.part(n).x
#         y = landmarks.part(n).y
#         landmark_tuple.append((x, y))
#         cv2.circle(img, (x, y), 2, (255, 255, 0), -1)

# Load the detector
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
specs_ori = cv2.imread('balaclava_rgba.png', -1)


def create_rgba():
    global specs_ori, h, w
    specs_ori = cv2.cvtColor(specs_ori, cv2.COLOR_BGR2BGRA)
    (h, w, _) = specs_ori.shape
    for i in range(h):
        for j in range(w):
            if specs_ori[i][j][0] != 0:
                specs_ori[i][j][3] = 0
    cv2.imwrite("new_glasses.png", specs_ori)


# create_rgba()
cap = cv2.VideoCapture(0)  # webcame video
# cap = cv2.VideoCapture('jj.mp4') #any Video file also
# cap.set(cv2.CAP_PROP_FPS, 30)


def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    mask = np.zeros(src.shape)
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image
    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
            if overlay[i][j][0] != 255:
                mask[x + i][y + j] = 255
    return src, mask


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


j = 0
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
# for i in range(len(dir) - 20000):
while True:
    ret, img = cap.read()
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


        for n in range(0, 81):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_tuple.append((x, y))
            cv2.circle(img, (x, y), 2, (255, 255, 0), -1)

        if isFrontal(landmarks, 0, 15):
            if mask is not None:
                cv2.imwrite("labels/" + str(j) + ".png", mask * 255)
                cv2.imwrite("masked/" + str(j) + ".png", img)
                cv2.imwrite("no_mask/" + str(j) + ".png", init)
                cv2.imshow('img.jpg', img)
            j = j + 1


    # if mask is not None:
    #     cv2.imwrite("CelebAMask-HQ/CelebA-HQ-img-256-256-labels/" + dir[i], mask * 255)
    #     cv2.imwrite("CelebAMask-HQ/CelebA-HQ-img-256-256-masked/" + dir[i], img)

    # img = np.round(model.predict(np.array([img]))[0, :, :, 0] * 255).astype("uint8")
    # img = cv2.resize(img, dsize=(480, 640))
    cv2.imshow('img.jpg', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


    # if mask_index != 5:
    #     mask_index = mask_index + 1
    # else:
    #     mask_index = 1

cap.release()
cv2.destroyAllWindows()
