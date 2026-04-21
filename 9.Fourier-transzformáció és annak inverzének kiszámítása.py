import numpy as np
import cv2

def dft2d(image):
    M, N = image.shape
    u = np.arange(M).reshape((M, 1))
    m = np.arange(M)
    W_M = np.exp(-2j * np.pi * u * m / M)

    v = np.arange(N).reshape((N, 1))
    n = np.arange(N)
    W_N = np.exp(-2j * np.pi * v * n / N)

    F = W_M @ image @ W_N.T
    return F.real, F.imag


def idft2d(real, imag):
    M, N = len(real), len(real[0])
    F = np.array(real) + 1j * np.array(imag)

    u = np.arange(M).reshape((M, 1))
    m = np.arange(M)
    W_M_inv = np.exp(2j * np.pi * u * m / M)

    v = np.arange(N).reshape((N, 1))
    n = np.arange(N)
    W_N_inv = np.exp(2j * np.pi * v * n / N)

    img_rec = (W_M_inv @ F @ W_N_inv.T) / (M * N)
    return img_rec.real

image = np.linspace(0, 255, 225).reshape(15, 15).astype(np.uint8)

real, imag = dft2d(image)

magnitude = np.sqrt(np.power(real, 2) + np.power(imag, 2))

magnitude_visible = np.log(1 + magnitude)
magnitude_visible = cv2.normalize(magnitude_visible, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

reconstructed_image = idft2d(real, imag)

reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

cv2.imwrite("fourier_spectum.pgm", magnitude_visible)
cv2.imwrite("reconstructed.pgm", reconstructed_image)

print("Eredeti első pixel:", image[0,0])
print("Visszaállított első pixel:", reconstructed_image[0,0])