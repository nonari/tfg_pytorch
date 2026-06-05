import biomarkers.extraction as ext
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imfile = "/home/nonari/Documentos/test_results/resnet34_cp_sa/seg_9_32.png"

im = cv.imread(imfile, cv.IMREAD_GRAYSCALE)

ext.rearrange_mask(im)

#ext.fill_gaps(im, 1)

fluid = ext.fluid_stats(im)

print(fluid)

layers = ext.layers_info(im)

print(layers)

proportion = sum(fluid['area_by_zone']) / (layers['retina_mean'] * im.shape[0])
print(proportion)
