import cv2
import numpy as np


def frequency_filter(img, d0, mode='low'):

    dft = np.fft.fftshift(cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT))

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    dist = np.sqrt(x * x + y * y)

    if mode == 'low':
        mask = dist <= d0
    else:  # high pass
        mask = dist > d0

    mask_stack = np.repeat(mask[:, :, np.newaxis], 2, axis=2)

    f_shift = dft * mask_stack
    img_back = cv2.idft(np.fft.ifftshift(f_shift))
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

img = cv2.imread("lena.pgm", 0)
cv2.imwrite("10Low.pgm",frequency_filter(img, 30, 'low'))
cv2.imwrite("10High.pgm",frequency_filter(img, 30, 'high'))