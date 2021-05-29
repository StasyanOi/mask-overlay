import os

import cv2
import numpy as np


def sort_names(dir):
    ints = []
    for i in range(len(dir)):
        ints.append(int(dir[i].split(".")[0]))
    ints.sort()
    for i in range(len(dir)):
        dir[i] = str(ints[i]) + ".png"
    return dir


def load_face_pictures_list_no_brightness(dir, lst, color_mode='grayscale'):
    images = []

    mode = -1
    if color_mode == 'grayscale':
        mode = 0
    else:
        mode = 1

    for i in range(0, len(lst)):
        input_arr_feature = cv2.imread(dir + "/" + lst[i], mode)
        if mode == 0:
            input_arr_feature = np.resize(input_arr_feature, (256, 256, 1))
        elif mode == 1:
            input_arr_feature = np.resize(input_arr_feature, (256, 256, 3))
        images.append(input_arr_feature)

    batch_feature = np.array(images)  # Convert single image to a batch.
    return batch_feature, lst


def merge_feature_mask(masked_people="./CelebAMask-HQ/CelebA-HQ-img-256-256-masked",
                       binary_labels="./CelebAMask-HQ/CelebA-HQ-img-256-256-labels",
                       merged_dir="./CelebAMask-HQ/CelebA-HQ-img-256-256-merged"):
    masked = masked_people + "/"
    img_labels = binary_labels + "/"
    merged = merged_dir + "/"

    dir_list = sort_names(os.listdir(masked))

    indexes = np.arange(0, 20000, 1)

    for p in range(len(indexes) - 1):
        files = [dir_list[i] for i in range(indexes[p + 1])]
        features, f_list = load_face_pictures_list_no_brightness(masked, files, color_mode='rgb')
        labels, l_list = load_face_pictures_list_no_brightness(img_labels, files)

        for i in range(len(features)):
            for j in range(features[i].shape[0]):
                for k in range(features[i].shape[1]):
                    if labels[i][j, k] == 255:
                        features[i][j, k, 0] = 255
                        features[i][j, k, 1] = 255
                        features[i][j, k, 2] = 255
            cv2.imwrite(merged + f_list[i], features[i].astype('uint8'))


if __name__ == '__main__':
    merge_feature_mask()
