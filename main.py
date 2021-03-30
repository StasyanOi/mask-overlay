import cv2
import dlib
import os
import numpy as np

# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

img = cv2.imread("crop/000115.jpg")
faces = face_detector(img, 1)

landmark_tuple = []
for k, d in enumerate(faces):
    landmarks = landmark_detector(img, d)
    for n in range(0, 81):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_tuple.append((x, y))
        cv2.circle(img, (x, y), 2, (255, 255, 0), -1)

# Load the detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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
cap.set(cv2.CAP_PROP_FPS, 30)


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


dir = os.listdir("./crop")
for i in range(len(dir)):
    ret, img = cap.read()
    # cv2.imwrite("not_masked/" + str(i) + ".png", img)
    # ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img)
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:
            symin = int(y + 0.0 * h / 8) - 20
            symax = int(y + 4.0 * h / 8) + 50
            y_diff = symax - symin
            right = 3
            sxmin = int(x + 0.0 * w / 8) + right
            sxmax = int(x + 7.5 * w / 8) + right
            x_diff = sxmax - sxmin
            face_glass_roi_color = img[symin:symax, sxmin:sxmax]
            specs = cv2.resize(specs_ori, (x_diff, y_diff), interpolation=cv2.INTER_CUBIC)
            _, mask = transparentOverlay(face_glass_roi_color, specs)
            w_mask = np.zeros(gray.shape)
            w_mask[symin:symax, sxmin:sxmax] = mask[:, :, 0]
    # cv2.imwrite("masked/" + str(i) + ".png", img)
    # cv2.imwrite("labels/" + str(i) + ".png", w_mask)
    cv2.imshow('img.jpg', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cv2.imwrite('img.jpg', img)
        break
cap.release()
cv2.destroyAllWindows()
