import cv2, numpy as np

img = cv2.imread("lena.pgm", 0)

kx = np.array([[-1, 0, 1]] * 3)
ky = kx.T

gx = cv2.filter2D(img, cv2.CV_32F, kx)
gy = cv2.filter2D(img, cv2.CV_32F, ky)

mag = np.hypot(gx, gy)
edges = (mag > 50).astype(np.uint8) * 255

cv2.imshow("Prewitt", edges)
cv2.waitKey(0)