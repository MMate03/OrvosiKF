import numpy as np

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

def histogram_equalization(img):
    hist = np.zeros(256, dtype=int)
    for pixel in img.flatten():
        hist[pixel] += 1

    total_pixels = img.size
    pdf = hist / total_pixels
    cdf = np.cumsum(pdf)
    cdf_normalized = np.round(cdf * 255).astype(np.uint8)
    equalized_img = cdf_normalized[img]
    return equalized_img

saveImage("1.pgm",histogram_equalization(loadImage("lena.pgm")))