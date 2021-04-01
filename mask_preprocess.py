import cv2
import dlib
import os
import csv
import numpy as np
import math


def create_rgba(img, m):
    specs_ori = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    (h, w, _) = specs_ori.shape
    for i in range(h):
        for j in range(w):
            if (specs_ori[i][j][0] > 254) & (specs_ori[i][j][1] > 254) & (specs_ori[i][j][2] > 254):
                specs_ori[i][j][3] = 0
    cv2.imwrite("masks/" + str(m) + "_new.png", specs_ori)


if __name__ == '__main__':
    for i in range(30000):
        img = cv2.imread("CelebAMask-HQ/CelebA-HQ-img/" + str(i) + ".jpg")
        img = cv2.resize(img, dsize=(256, 256))
        cv2.imwrite("CelebAMask-HQ/CelebA-HQ-img-256-256/" + str(i) + ".png", img)
    # i = 5
    # first = cv2.imread("masks/" + str(i) + ".png")
    # create_rgba(first, i)
