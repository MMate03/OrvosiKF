import cv2
import numpy as np

def gray_morph(img, kernel_size=3, op='dilate'):
    padded = np.pad(img, kernel_size // 2, mode='edge')

    stack = [padded[i: i + img.shape[0], j: j + img.shape[1]]
             for i in range(kernel_size) for j in range(kernel_size)]

    return np.maximum.reduce(stack) if op == 'dilate' else np.minimum.reduce(stack)

img = cv2.imread("lena.pgm", 0)

dilated = gray_morph(img, 5, 'dilate')
cv2.imwrite("8.dilated.pgm", dilated)

eroded = gray_morph(img, 5, 'erode')
cv2.imwrite("8.eroded.pgm", eroded)