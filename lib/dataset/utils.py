import cv2
import numpy as np
from skimage.filters import threshold_otsu
import skimage.transform
import os
import time
import torch
import glob
import pyfftw
import multiprocessing


def get_pixels_on_line(image, alpha, h, c, num_of_points, point_number_fraction=1.0):
    ys = np.linspace(c[0] + np.cos(alpha * 0.01745) * (-h / 8), c[0] + np.cos(alpha * 0.01745) * (h / 8), num_of_points,
                     dtype=np.int32)
    xs = np.linspace(c[1] - np.sin(alpha * 0.01745) * (-h / 8), c[1] - np.sin(alpha * 0.01745) * (h / 8), num_of_points,
                     dtype=np.int32)
    return ys, xs


def deskew_fft(input_image, angle_range=10, step=0.25, point_number_fraction=1.0, downsample_constant=1.0,
               crop_percent=0.0):
    image = input_image.copy()

    if downsample_constant > 1.0:
        image = cv2.resize(image,
                           (int(image.shape[1] // downsample_constant), int(image.shape[0] // downsample_constant)))
    if crop_percent > 0.0:
        x = int(image.shape[1] * (crop_percent / 100.0))
        y = int(image.shape[0] * (crop_percent / 100.0))
        image = image[y:-y, x:-x]
        # $print(image.shape)
    h = image.shape[0]
    w = image.shape[1]
    num_of_points = int(np.floor(np.sqrt((h / 4) ** 2 + (w / 4) ** 2) * point_number_fraction))
    mult_coef = 1 / step
    start_fft = time.time()
    aa = pyfftw.byte_align(image, dtype='complex64')
    af = pyfftw.empty_aligned(image.shape, dtype='complex64')
    fft_object_c = pyfftw.FFTW(aa, af, axes=(0, 1), flags=('FFTW_ESTIMATE',), direction='FFTW_FORWARD',
                               threads=multiprocessing.cpu_count())
    ft = fft_object_c(aa)
    ftshift = np.fft.fftshift(ft)
    stop_fft = time.time()
    c = [h // 2, w // 2]
    spek = 20 * np.log(np.abs(ftshift[c[0] - h // 7: c[0] + h // 7, c[1] - w // 7: c[1] + w // 7]))
    c2 = [spek.shape[0] / 2, spek.shape[1] / 2]
    angles = []
    start_angle = time.time()

    angle_list = [-angle_range + i * step for i in range(0, 2 * int(angle_range * mult_coef + 1))]
    pixel_array = np.zeros((len(angle_list), num_of_points))

    for i, alpha in enumerate(angle_list):
        angles.append(alpha)
        ys, xs = get_pixels_on_line(spek, alpha, h, c2, num_of_points, point_number_fraction)
        pixel_array[i, :] = spek[ys, xs]

    stop_angle = time.time()
    idx = np.argmax(np.sum(np.abs(pixel_array), axis=1))
    # print(f"FFT time: {stop_fft - start_fft}, ANGLE time {stop_angle - start_angle}, IDX {idx}")
    angle = angle_list[idx]
    if angle == -angle_range:
        angle = 0.0

    return angle


def rgb_binary(image):
    if len(image.shape) != 3:
        return False
    thresh_r = threshold_otsu(image[:, :, 0])
    thresh_g = threshold_otsu(image[:, :, 1])
    thresh_b = threshold_otsu(image[:, :, 2])
    binary_r = image[:, :, 0] > thresh_r
    binary_g = image[:, :, 1] > thresh_g
    binary_b = image[:, :, 2] > thresh_b
    binary = np.dstack([binary_r, binary_g, binary_b])
    result = np.uint8(np.sum(binary, axis=2) > 0) * 255

    return result


def rotate(image, angle):
    if angle == 0:
        return image
    else:
        return skimage.transform.rotate(image, angle, preserve_range=True, resize=True)


def rotate_slow(image, angle):
    temp_image = np.ones([image.shape[0] * 2, image.shape[1] * 2], dtype=np.uint8) * np.mean(image)
    ymin = int(temp_image.shape[0] / 2.0 - image.shape[0] / 2.0)
    ymax = int(temp_image.shape[0] / 2.0 + image.shape[0] / 2.0)
    xmin = int(temp_image.shape[1] / 2.0 - image.shape[1] / 2.0)
    xmax = int(temp_image.shape[1] / 2.0 + image.shape[1] / 2.0)
    temp_image[ymin: ymax, xmin: xmax] = image
    temp_image = skimage.transform.rotate(temp_image, angle)
    image = temp_image[ymin: ymax, xmin: xmax]

    return image


def crop_image(img, tol=0):
    # img is 2D image data
    # tol  is tolerance
    if len(img.shape) == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    else:
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > tol
        return img[np.ix_(mask.any(1), mask.any(0))]


def pad_image(img, aspect_ratio=1.0):
    """
    Pads an image to desired ratio.

    Args:
        img: input image
        aspect_ratio: desired width-to-height ratio

    Returns:
        Padded image.

    """
    longer_side = np.maximum(img.shape[0], img.shape[1])

    if len(img.shape) == 2:
        canvas = np.zeros((longer_side, np.floor(longer_side * aspect_ratio).astype(int)), dtype=np.uint8)
    else:
        canvas = np.zeros((longer_side, np.floor(longer_side * aspect_ratio).astype(int), img.shape[-1]),
                          dtype=np.uint8)

    delta_x = (canvas.shape[1] - img.shape[1]) // 2
    delta_y = (canvas.shape[0] - img.shape[0]) // 2

    canvas[delta_y:delta_y + img.shape[0], delta_x:delta_x + img.shape[1]] = img

    return canvas


def crop_and_deskew(image):
    image = crop_image(image, tol=75)
    src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(src, (5, 5))
    angle = deskew_fft(blur, step=0.25, downsample_constant=3, angle_range=25, point_number_fraction=1.0,
                       crop_percent=10)
    image = rotate(image, angle).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image(image, 50)
    image = pad_image(image)

    return image


class CropAndDeskew:
    def __init__(self):
        pass

    def __call__(self, image):
        # return torch.from_numpy(crop_and_deskew(image))
        return crop_and_deskew(image)


class PadImage:
    def __init__(self):
        pass

    def __call__(self, image, aspect_ratio=1.0):
        # return torch.from_numpy(pad_image(image, aspect_ratio))
        return pad_image(image, aspect_ratio)


if __name__ == "__main__":
    files = sorted(glob.glob("/home/neduchal/Dokumenty/Data/Naki/deskew_set/*.jpg"))
    print(files)
    for i, f in enumerate(files):
        src = cv2.imread(f, 0)
        blur = cv2.blur(src, (5, 5))
        img = src.copy()
        start = time.time()
        angle = deskew_fft(blur, step=0.25)
        print(os.path.basename(f), angle, time.time() - start)
        cv2.imwrite("../data/output/pyfftw/" + os.path.basename(f), crop_image(rotate(src, angle), 50))
        if i > 4:
            break
