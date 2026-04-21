import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def saveImage(filename, img):
    with open(filename, 'wb') as f:
        header = f"P5\n{img.shape[1]} {img.shape[0]}\n255\n"
        f.write(header.encode())
        f.write(img.tobytes())

def loadImage(filename):
    with open(filename, 'rb') as f:

        header_raw = f.read().split(None, 4)
        magic, w, h, mx, data = header_raw

        if magic == b'P5':
            img = np.frombuffer(data, dtype=np.uint8)
        else:
            img = np.fromstring(data, sep=' ', dtype=np.uint8)

    return img.reshape((int(h), int(w)))

def median_filter(img, k=3):
    padded = np.pad(img, k // 2, mode='edge')
    windows = sliding_window_view(padded, (k, k))

    return np.median(windows, axis=(2, 3)).astype(np.uint8)

saveImage("3.pgm",median_filter(loadImage("lena.pgm")))