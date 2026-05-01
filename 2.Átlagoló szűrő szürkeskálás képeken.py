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


def average_filter(img, kernel_size=3):
    res = np.zeros(img.shape, dtype=float)
    padded = np.pad(img, kernel_size // 2, mode='edge')

    for i in range(kernel_size):
        for j in range(kernel_size):

            res += padded[i:i + img.shape[0], j:j + img.shape[1]]

    return (res / (kernel_size ** 2)).astype(np.uint8)



saveImage("2.pgm", average_filter(loadImage("lena.pgm")))