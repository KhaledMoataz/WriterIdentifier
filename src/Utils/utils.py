#!/usr/bin/env python3

import cv2
import numpy as np


def read_image(path, scale=30):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Reduce the image size
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def show_images(images, names=None):
    for i in range(len(images)):
        name = str(i + 1)
        if names is not None:
            name = names[i]
        cv2.imshow(name, images[i])
    cv2.waitKey(0)


def read_ascii(path="../data/ascii/forms.txt"):
    f = open(path, "r")
    writers_list = [[] for _ in range(672)]
    for line in f:
        if line[0] == '#':
            continue

        img_name, writer, a, b, c, d, e, f = line.split()
        if img_name[0] == 'e':
            break
        writer = int(writer)
        writers_list[writer].append(img_name)

    return writers_list


def get_random_indices(original_list, number):
    random_list = []
    for i in range(number):
        j = 0
        while True:
            idx = np.random.randint(len(original_list))
            j += 1
            if idx not in random_list and len(original_list[idx]) > 1:
                random_list.append(idx)
                break
            if j > 10:
                return random_list
    return random_list
