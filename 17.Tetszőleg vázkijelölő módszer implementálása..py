import cv2, numpy as np

img = cv2.imread("lena.pgm", 0)
_, bin = cv2.threshold(img, 127, 255, 0)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
skel = np.zeros_like(bin)

while cv2.countNonZero(bin):
    eroded = cv2.erode(bin, kernel)
    # A váz része: az aktuális kép és annak nyitása közötti különbség
    temp = cv2.subtract(bin, cv2.dilate(eroded, kernel))
    skel = cv2.bitwise_or(skel, temp)
    bin = eroded

cv2.imshow("Vaz", skel)
cv2.waitKey(0)