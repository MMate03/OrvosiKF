import cv2, numpy as np

img = cv2.imread("lena.pgm", 0)
edges = cv2.Canny(cv2.GaussianBlur(img, (5, 5), 0), 100, 200)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

if lines is not None:

    dx = lines[:, 0, 2] - lines[:, 0, 0]
    dy = lines[:, 0, 3] - lines[:, 0, 1]

    angles = np.degrees(np.arctan2(dy, dx)) % 180
    lengths = np.sqrt(dx ** 2 + dy ** 2)

    hist, bin_edges = np.histogram(angles, bins=180, range=(0, 180), weights=lengths)

    dominant_angle = bin_edges[np.argmax(hist)]
    print(f"Domináns irány: {dominant_angle:.2f} fok")