import os
import numpy as np
import cv2.cv2 as cv2

def get_mean_face(images):
    mean_face = images[0].astype('float64')

    stop = len(images)
    for i in range(1, stop):
        mean_face = mean_face + images[i].astype('float64')

    return (mean_face / stop).astype('uint8')

def get_variance(images, mean):
    mean = mean.astype('float64')
    images = images.astype('float64')
    for i in range(0, len(images)):
        images[i] = images[i] - mean
        images[i] = (images[i] - images[i].min()) / (images[i].max() - images[i].min()) * 255

    return images

if __name__ == '__main__':
    celeb_dir = "../CelebAMask-HQ/CelebA-HQ-img-256-256"
    images = os.listdir(celeb_dir)

    faces = []
    for i in range(100):
        image = cv2.imread(celeb_dir + "/" + images[i], cv2.IMREAD_GRAYSCALE)
        faces.append(image)

    stack = np.stack(faces)

    mean_face = get_mean_face(stack)

    eugenfaces = get_variance(stack, mean_face)

    cv2.imwrite("../eugen/mean_face.png", mean_face)

    for i in range(len(eugenfaces)):
        cv2.imwrite("eugen/" + str(i) + ".png", eugenfaces[i].astype('uint8'))

