import cv2
import numpy as np


def quick_dilate(img, k=3):

    pad = k // 2
    padded = np.pad(img, pad, mode='edge')

    stack = [padded[i: i + img.shape[0], j: j + img.shape[1]]
             for i in range(k) for j in range(k)]

    return np.maximum.reduce(stack)

img = cv2.imread("lena.pgm", 0)
cv2.imwrite("6.pgm", quick_dilate(img, 7))