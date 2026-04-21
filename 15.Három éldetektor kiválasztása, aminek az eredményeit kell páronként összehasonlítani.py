import cv2, numpy as np

img = cv2.imread("lena.pgm", 0)
img = cv2.GaussianBlur(img, (5, 5), 0)

canny = cv2.Canny(img, 100, 200)

sx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobel = cv2.convertScaleAbs(np.hypot(sx, sy))

lap = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F))

detectors = {"Canny": canny, "Sobel": sobel, "Laplacian": lap}
names = list(detectors.keys())
items = list(detectors.values())

for name, res in detectors.items():
    cv2.imshow(name, res)

for i in range(3):
    for j in range(i + 1, 3):
        diff = cv2.absdiff(items[i], items[j])
        diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow(f"{names[i]} vs {names[j]}", diff)

cv2.waitKey(0)