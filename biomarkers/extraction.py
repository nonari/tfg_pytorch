import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def rearrange_mask(mask):
    sparse_labels = np.unique(mask)
    for idx, l in enumerate(sparse_labels):
        mask[mask == l] = idx + 1


def fill_gaps(im, layer):
    mask = im.copy()
    mask[mask != layer] = 0
    stats = cv.connectedComponentsWithStats(im, connectivity=8)
    vols = stats[2][1:, 1]
    obj_idx = np.argmin(vols) + 1
    res = stats[1]
    res[res != obj_idx] = 0
    res = res.astype(np.uint8)
    conts, hier = cv.findContours(res, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    filled_gaps = np.zeros(im.shape)
    cv.drawContours(filled_gaps, conts, -1, 1, thickness=cv.FILLED)
    im[filled_gaps == 1] = layer


def layer_thickness(im, layer):
    mask = im.copy()
    mask[mask != layer] = 0
    mask[mask == layer] = 1
    arrow = np.arange(0, mask.shape[1])
    arrow = arrow.reshape(mask.shape[1], 1)
    hmap = arrow * mask
    hmap[hmap == 0] = 999999
    upper = np.argmin(hmap, axis=0)
    hmap[hmap == 999999] = -1
    lower = np.argmax(hmap, axis=0)
    diff_layer = lower - upper + 1
    return diff_layer


def mask_retina(im):
    im = im.copy()
    im[im == 9] = 0
    im[im == 1] = 0
    im[im == 10] = 1
    for i in range(2, 9):
        im[im == i] = 1
    return im


def fluid_stats(im):
    im = im.copy()
    im[im != 10] = 0
    im[im == 10] = 1

    fluid_data = {}
    stats = cv.connectedComponentsWithStats(im)
    areas = stats[2][1:, 4]
    total_count = len(areas)
    fluid_data['total'] = total_count
    sep = [0, 0.2, 0.4, 0.6, 0.8, 1]
    centroids = stats[3][1:, 0]

    x, y = im.shape
    areas_zone = []
    count_zone = []

    for i in range(5):
        p1, p2 = sep[i], sep[i + 1]
        ini, fin = int(p1 * x), int(p2 * x)
        area = (im[:, ini:fin] == 1).sum()
        areas_zone.append(area)
        count_l = ini < centroids
        count_g = fin > centroids
        count = np.logical_and(count_l, count_g).sum()
        count_zone.append(count)

    fluid_data["area_by_zone"] = areas_zone
    fluid_data["count_by_zone"] = count_zone
    conts, hier = cv.findContours(im, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    by_cyst = {}
    fluid_data['by_cyst'] = by_cyst
    for c in conts:
        _, _, w, h = cv.boundingRect(c)
        x, y = c[0, 0]
        idx = stats[1][y, x]
        by_cyst[idx] = {"measures": (w, h)}
        by_cyst[idx]["ratio"] = round(h / w, 3)
        by_cyst[idx]["area"] = stats[2][idx, 4]

    return fluid_data


def layers_info(im):
    layers = {}
    by_layer = {}
    layers["by_layer"] = by_layer
    for l_idx in range(2, 9):
        thickness = layer_thickness(im, l_idx)
        std = round(np.std(thickness), 3)
        mean = round(np.mean(thickness), 3)
        maxim = np.max(thickness)
        minim = np.min(thickness)
        by_layer[l_idx] = {"std": std}
        by_layer[l_idx]['mean'] = mean
        by_layer[l_idx]['max'] = maxim
        by_layer[l_idx]['min'] = minim

    retina_mask = mask_retina(im)
    thickness = layer_thickness(retina_mask, 1)
    std = np.std(thickness)
    mean = np.mean(thickness)
    maxim = np.max(thickness)
    minim = np.min(thickness)
    layers['retina_std'] = std
    layers['retina_mean'] = mean
    layers['retina_max'] = maxim
    layers['retina_min'] = minim

    return layers
