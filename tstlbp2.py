import time

from lbp2 import rlbp, disk_shape, rlbp_short
import matplotlib.pyplot as plt
import cv2
import numpy as np

mask, P = disk_shape(3, (3,3), (16,16))
mask[3, 3] = False
mask = np.where(mask)

for i in range(1):
    img_path = f'/home/nonari/Documentos/itg/vehicle/0a0a00b2fbe89a47.jpg'

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    t1 = time.time()


    #rlbp_s, rlbp_m = rlbp(img, mask, P, alpha=0.2, threshold=np.mean(img))
    rlbp_s, rlbp_m = rlbp_short(img, mask, alpha=0.2, threshold=np.mean(img))
    print("XXX", time.time() - t1)
    cv2.imshow("s", rlbp_s)
    cv2.imshow("m", rlbp_m)
    cv2.waitKey(0)

'''
fig, ax = plt.subplots(1,3)
ax = ax.ravel()

ax[0].imshow(img, cmap='gray')
ax[1].imshow(rlbp_s, cmap='gray')
ax[2].imshow(rlbp_m, cmap='gray')

plt.show()
'''
