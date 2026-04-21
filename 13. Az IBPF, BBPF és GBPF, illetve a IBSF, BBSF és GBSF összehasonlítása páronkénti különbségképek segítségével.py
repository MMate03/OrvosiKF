import cv2, numpy as np

img = cv2.imread("lena.pgm", 0).astype(float)
h, w = img.shape
f = np.fft.fftshift(np.fft.fft2(img))
y, x = np.ogrid[-h//2:h-h//2, -w//2:w-w//2]
D = np.sqrt(x*x + y*y)

d0, d1, n = 30, 60, 2
W, Dmid = d1 - d0, (d1 + d0) / 2
eps = 1e-9

IBP = (D >= d0) & (D <= d1)

BBP = 1 / (1 + (D * W / (D**2 - Dmid**2 + eps) + eps)**(2 * n))
GBP = np.exp(-((D**2 - Dmid**2) / (D * W + eps))**2)

M = [IBP, BBP, GBP]
M += [1 - m for m in M]
N = ["IBP","BBP","GBP","IBS","BBS","GBS"]

res = []
for i in range(6):
    r = np.abs(np.fft.ifft2(np.fft.ifftshift(f * M[i]))).astype(np.uint8)
    cv2.imshow(N[i], r)
    res.append(r)

for i, j in [(0,1),(0,2),(1,2),(3,4),(3,5),(4,5)]:
    d = cv2.absdiff(res[i], res[j])
    cv2.imshow(f"D_{N[i]}_{N[j]}", cv2.normalize(d, None, 0, 255, 32))

cv2.waitKey(0)