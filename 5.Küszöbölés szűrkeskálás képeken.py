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

def thresholding(img, threshold=128):
    binary_img = np.zeros_like(img)
    binary_img[img > threshold] = 255
    return binary_img

saveImage("5.pgm", thresholding(loadImage("lena.pgm")))