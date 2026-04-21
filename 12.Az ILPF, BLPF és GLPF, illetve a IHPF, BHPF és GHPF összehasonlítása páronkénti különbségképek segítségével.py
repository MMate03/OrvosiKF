import cv2, numpy as np

img = cv2.imread("lena.pgm", 0).astype(float)
h, w = img.shape
f = np.fft.fftshift(np.fft.fft2(img))
y, x = np.ogrid[-h//2:h-h//2, -w//2:w-w//2]
D, D0, n = np.sqrt(x*x + y*y), 40, 2

L = [D<=D0, 1/(1+(D/D0)**(2*n)), np.exp(-(D**2)/(2*D0**2))]
M = L + [1-m for m in L]
N = ["IL","BL","GL","IH","BH","GH"]

res = []
for i in range(6):
    r = np.abs(np.fft.ifft2(np.fft.ifftshift(f * M[i]))).astype(np.uint8)
    cv2.imshow(N[i], r)
    res.append(r)

for i, j in [(0,1),(0,2),(1,2),(3,4),(3,5),(4,5)]:
    d = cv2.absdiff(res[i], res[j])
    cv2.imshow(f"D_{N[i]}_{N[j]}", cv2.normalize(d, None, 0, 255, 32))

cv2.waitKey(0)