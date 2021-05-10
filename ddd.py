import torch
import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Create mapping
# Get color codes for dataset (maybe you would have to use more than a single
# image, if it doesn't contain all classes)
im = cv.imread("/home/nonari/Documentos/tfgdata/seg_0.png")
img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
h, w = img.shape
colors = np.unique(img)
mask = torch.zeros(1, len(colors), h, w)
for idx, color in enumerate(colors):
    x, y = np.where(img == color)
    mask[0, idx, x, y] = 1
