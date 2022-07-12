import cv2 as cv
import numpy as np


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1],
                           flags=cv.INTER_LINEAR)
    return result


def random_flip(I):
    if np.random.random() >= 0.5:
        I = cv.flip(I, 1)
    if np.random.random() >= 0.7:
        I = cv.flip(I, 0)

    return I


def random_rotate(I):
    return rotate_image(I, np.round(np.random.random() * 45))


def random_rotate_and_flip(I):
    return random_rotate(random_flip(I))


def replace_image(I, mask, loc=(0, 0), w=0.1):
    mask *= w
    end = (np.minimum(I.shape[0], loc[0] + mask.shape[0]),
           np.minimum(I.shape[1], loc[1] + mask.shape[1]))
    I[loc[0]:end[0], loc[1]:end[1]] = I[loc[0]:end[0], loc[1]:end[1]] * (
            1 - w) + mask[:np.minimum(mask.shape[0], I.shape[0] - loc[0]),
                     :np.minimum(mask.shape[1], I.shape[1] - loc[1])]


def add_reflection(I, refs, ref_range=(0.5, 1.5), max_ref_count=6,
                   weight_range=(0.01, 0.4)):
    I = I.copy()
    start = (0, 0)
    end = (I.shape[0], I.shape[1])
    ref_indices = []
    for i in range(max_ref_count):
        ref_indices.append(
            np.floor(np.random.random() * len(refs)).astype("int"))
    at_least_one = False
    for ind in ref_indices:
        if at_least_one and np.random.random() > 0.7:
            continue
        at_least_one = True
        refl = refs[ind]
        scl = (ref_range[1] - ref_range[0]) * np.random.random() + ref_range[0]
        refl = resize(refl, scale=scl)
        refl = random_rotate_and_flip(refl)
        refl = blur(refl)
        randX = np.floor((end[0] - start[0]) * np.random.random() + start[0]).astype("int")
        randY = np.floor((end[1] - start[1]) * np.random.random() + start[1]).astype("int")
        w = (weight_range[1] - weight_range[0]) * np.random.random() + \
            weight_range[0]
        replace_image(I, refl, (randX, randY), w)
    return I


def generate_batch(I, refs, size=16):
    I1 = cv.flip(I.copy(), 1)  # other side of the road
    x = []
    y = []
    for i in range(size // 2):
        x.append(I)
        y.append(add_reflection(I, refs))
    for i in range(size // 2):
        x.append(I1)
        y.append(add_reflection(I1, refs))
    return x, y


def blur(img, k=5):
    return cv.GaussianBlur(img, (k, k), 0)


def resize(I, scale):
    return cv.resize(I, (
        np.round(I.shape[0] * scale).astype("int"), np.round(I.shape[1] * scale).astype("int")))
