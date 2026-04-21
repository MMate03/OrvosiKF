import numpy as np
import cv2


def hit_or_miss_ultra_short(image, kernel):
    h, w = image.shape
    pad = np.pad(image, (kernel.shape[0] // 2, kernel.shape[1] // 2), mode='constant')

    # Listakomprehenzióval összegyűjtjük a feltételeket, majd az összeset össze-ÉSeljük
    masks = [(pad[i:i + h, j:j + w] == kernel[i, j])
             for i in range(kernel.shape[0]) for j in range(kernel.shape[1]) if kernel[i, j] != -1]

    return np.all(masks, axis=0).astype(np.uint8)

img = cv2.imread("lena.pgm", 0)
binary_img = (img > 127).astype(np.uint8)

kernel = np.array([
    [ 0,  0, -1],
    [ 0,  1, -1],
    [-1, -1, -1]
])

result = hit_or_miss_ultra_short(binary_img, kernel)

cv2.imwrite("7.pgm", result * 255)