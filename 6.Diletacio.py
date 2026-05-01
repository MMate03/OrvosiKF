import cv2
import numpy as np


def dilate(img, kernel_size=3):


    padded = np.pad(img, kernel_size//2, mode='edge')

    stack = [padded[i: i + img.shape[0], j: j + img.shape[1]]
             for i in range(kernel_size) for j in range(kernel_size)]

    return np.maximum.reduce(stack)

img = cv2.imread("lena.pgm", 0)
cv2.imwrite("6.pgm", dilate(img, 7))