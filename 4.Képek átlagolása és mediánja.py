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


def combine_images(filenames, operation='mean'):
    images = [loadImage(f) for f in filenames]

    stack = np.array(images)

    if operation == 'mean':
        res = np.mean(stack, axis=0)
    else:
        res = np.median(stack, axis=0)

    return res.astype(np.uint8)


files = ["baboon.pgm", "lena.pgm"]

avg_res = combine_images(files, 'mean')
saveImage("4.sorozat_atlag.pgm", avg_res)


med_res = combine_images(files, 'median')
saveImage("4.sorozat_median.pgm", med_res)