import numpy as np
from scipy.ndimage import convolve
import itertools


def disk_shape(r, center, size):
    a, b = center
    w, h = size

    y, x = np.ogrid[-a:w - a, -b:h - b]
    mask = x * x + y * y <= r * r

    return mask, np.sum(mask)


def rlbp_s(wlg, pos, mask, image):
    neighbours = image[mask]
    sub = neighbours/255 - wlg[pos]
    sub[np.where(sub > 0)] = 1
    sub[np.where(sub < 0)] = 0
    idx = np.arange(len(sub))
    two = np.full(len(sub), 2)
    return np.sum((sub * two) ** idx)


def rlbp_m(wlg, pos, mask, thresh):
    neigh_sub_center = wlg[mask] - wlg[pos]
    m_c = neigh_sub_center - thresh
    m_c[np.where(m_c > 0)] = 1
    m_c[np.where(m_c < 0)] = 0
    idx = np.arange(len(m_c))
    two = np.full(len(m_c), 2)
    return np.sum((m_c * two) ** idx)

def vec_pulse(i):
    if i > 0:
        return 1
    else:
        return 0

def zero_shift(mat, h, v):
    r, c = mat.shape
    res_m = mat
    if h > 0:
        z = np.zeros((h, c))
        res_m = np.vstack((res_m, z))
        res_m = res_m[h:r+h, :]
    else:
        h = abs(h)
        z = np.zeros((h, c))
        res_m = np.vstack((z, res_m))
        res_m = res_m[0:r, :]

    if v > 0:
        z = np.zeros((r, v))
        res_m = np.hstack((res_m, z))
        res_m = res_m[:, v:c+v]
    else:
        v = abs(v)
        z = np.zeros((r, v))
        res_m = np.hstack((z, res_m))
        res_m = res_m[:, 0:c]

    return res_m

def wlgPrecalculateComp(img, alpha, mask, P):
    kernel = np.zeros((max(mask[0])+1, max(mask[1])+1))
    kernel[mask] = 1

    c_img = img.copy()
    c_img = np.ndarray.astype(c_img, np.float32)
    # c_img = c_img / 255

    neighbourhood = convolve(c_img, kernel, mode='nearest', cval=0)
    gci = neighbourhood - c_img

    res = (c_img * alpha + gci) / (alpha + P)

    # res_norm = (res - np.min(res)) / np.ptp(res)

    return res


def rlbp_s_mat(wlg, mask, img, zindexed):
    mask_dx = mask[0] - 2
    mask_dy = mask[1] - 2

    final = None
    for idx, (i, j) in enumerate(zip(mask_dx, mask_dy)):
        sh = zero_shift(img, j, i) - wlg
        sh[sh > 0] = 1
        sh[sh < 0] = 0
        part = (sh * 2) ** idx
        if final is None:
            final = part
        else:
            final = final + part

    return final


def rlbp_m_mat(wlg, mask, thresh, zindexed):
    mask_dx = mask[0] - 2
    mask_dy = mask[1] - 2

    final = None
    for idx, (i, j) in enumerate(zip(mask_dx, mask_dy)):
        zs = zero_shift(wlg, j, i)
        sh = np.abs(zs - wlg) - 0
        sh[sh > 0] = 1
        sh[sh < 0] = 0
        part = (sh * 2) ** idx
        if final is None:
            final = part
        else:
            final = final + part

    return final


def rlbp_short(img, mask, threshold, alpha):
    img = img.astype(np.int32)
    zindexed = np.arange(1, len(mask[0])+1).reshape((1, 1, len(mask[0])))

    wlg = wlgPrecalculateComp(img, alpha, mask, len(mask[0]))
    return rlbp_s_mat(wlg, mask, img, zindexed), rlbp_m_mat(wlg, mask, threshold, zindexed)


def rlbp_of(img, x, y, mask, threshold, wlg):
    posX, posY = (mask[0] + x - 3, mask[1] + y - 3)
    h, w = img.shape
    count = 0
    for ele in itertools.takewhile(lambda t: t[0] < 0 or t[1] < 0 or t[0] >= h or t[1] >= w, zip(posX, posY)):
        count = count + 1
    inbound = count == 0
    if not inbound:
        return 0, 0

    valS = rlbp_s(wlg, (x, y), (posX, posY), img)
    valM = rlbp_m(wlg, (x, y), (posX, posY), threshold)
    print(x, y)
    return valM, valS


def rlbp(img, mask, P, alpha=0.5, threshold=0):
    rlbp_m = np.zeros(img.shape)
    rlbp_s = np.zeros(img.shape)

    wlg = wlgPrecalculateComp(img, alpha, mask, P)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rlbp_m[i, j], rlbp_s[i, j] = rlbp_of(img, i, j, mask, threshold, wlg)

    return rlbp_s, rlbp_m


ss = np.ones((2, 2))
zero_shift(ss, -1, -1)
