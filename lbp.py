import numpy as np
import cv2
import sys


def disk_shape(r, center, size):
    a, b = center
    w, h = size

    y, x = np.ogrid[-a:w - a, -b:h - b]
    mask = x * x + y * y <= r * r

    return mask, np.sum(mask)


def s(v):
    if v >= 0:
        return 1
    else:
        return 0


def get_pixel(img, pos, wlg_val):
    val = 0
    try:
        val = s(img[pos] - wlg_val)
    except:
        pass

    return val


def wlg(img, alpha, x, y, mask, P):
    center = img[x, y]

    gci = np.sum(img[mask] - center)

    return (center * alpha + gci) / (alpha + P)


def mp(img, alpha, x, y, wlgPrecalculate, c1, c2, threshold, mask, P):
    if wlgPrecalculate[c1, c2] == 0:
        wlgPrecalculate[c1, c2] = wlg(img, alpha, c1, c2, mask, P)

    m = np.abs(wlgPrecalculate[x, y] - wlgPrecalculate[c1, c2])

    return s(m - threshold)


def rlbp_of(img, x, y, mask, P, alpha, threshold, wlgPrecalculate):
    center = img[x, y]

    valM = 0
    valS = 0
    pxM = 0
    pxS = 0

    power_val = 0

    posX, posY = (mask[0] + x, mask[1] + y)

    if wlgPrecalculate[x, y] == 0:
        wlgPrecalculate[x, y] = wlg(img, alpha, x, y, (posX, posY), P)

    for i in range(P):
        if posX[i] >= img.shape[0] or posY[i] >= img.shape[1]:
            continue
        if (posX[i], posY[i]) == (x, y):
            # Pasamos do centro
            continue

        pxS = get_pixel(img, (posX[i], posY[i]), wlgPrecalculate[x, y])
        pxM = mp(img, alpha, x, y, wlgPrecalculate, posX[i], posY[i], threshold, mask, P)
        valM += pxM * (2 ** power_val)
        valS += pxS * (2 ** power_val)

        power_val += 1

    return valM, valS


def rlbp(img, mask, P, alpha=0.5, threshold=0):
    rlbp_m = np.zeros(img.shape)
    rlbp_s = np.zeros(img.shape)
    wlgPrecalculate = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rlbp_m[i, j], rlbp_s[i, j] = rlbp_of(img, i, j, mask, P, alpha, threshold, wlgPrecalculate)

    return rlbp_s, rlbp_m