import cv2
import numpy as np

def band_filter(img, d0, d1, mode='pass'):

    dft = np.fft.fftshift(cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT))

    r, c = img.shape
    y, x = np.ogrid[-r // 2:r - r // 2, -c // 2:c - c // 2]
    dist = np.sqrt(x * x + y * y)

    mask = (dist >= d0) & (dist <= d1)

    if mode == 'stop':
        mask = ~mask

    mask_stack = np.repeat(mask[:, :, np.newaxis], 2, axis=2)

    f_shift = dft * mask_stack
    img_back = cv2.idft(np.fft.ifftshift(f_shift))
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

img = cv2.imread("lena.pgm", 0)
# Csak a 20 és 50 közötti frekvenciák maradnak meg
cv2.imwrite("11Pass.pgm",band_filter(img, 20, 50, 'pass'))
# A 20 és 50 közötti frekvenciákat kivágjuk
cv2.imwrite("11Stop.pgm",band_filter(img, 20, 50, 'stop'))