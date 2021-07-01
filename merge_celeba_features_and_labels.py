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


def load_face_pictures(dir, lst, color_mode='grayscale'):
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


def merge_feature_mask(masked="./mask_overlay_datasets/CelebA-HQ-img-256-256-masked",
                       binary="./mask_overlay_datasets/CelebA-HQ-img-256-256-labels",
                       merged="./mask_overlay_datasets/CelebA-HQ-img-256-256-merged"):
    masked_dir = masked + "/"
    masked_labels_dir = binary + "/"
    merged_dir = merged + "/"

    images_to_merge = sort_names(os.listdir(masked_dir))

    total_num_of_images_to_merge = len(images_to_merge)
    indexes = np.arange(0, total_num_of_images_to_merge, 100)
    indexes = np.append(indexes, total_num_of_images_to_merge)

    for p in range(len(indexes) - 1):
        files = [images_to_merge[i] for i in range(indexes[p], indexes[p + 1])]
        features, f_list = load_face_pictures(masked_dir, files, color_mode='rgb')
        labels, l_list = load_face_pictures(masked_labels_dir, files)
        for i in range(len(features)):
            features_temp = features[i]
            label_temp = cv2.cvtColor(labels[i], cv2.COLOR_GRAY2RGB)
            inverted = np.invert(label_temp[:, :, 0])
            uint_inverted = (inverted / 255).astype('uint8')
            features_temp[:, :, 0] = features_temp[:, :, 0] * uint_inverted
            features_temp[:, :, 1] = features_temp[:, :, 1] * uint_inverted
            features_temp[:, :, 2] = features_temp[:, :, 2] * uint_inverted
            features_temp = features_temp + label_temp
            cv2.imwrite(merged_dir + f_list[i], features_temp.astype('uint8'))


if __name__ == '__main__':
    merge_feature_mask()
