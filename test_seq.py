from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from os import path
import os
import cv2
path_to_dataset = "/home/nonari/Descargas/2015_BOE_Chiu/"
for dirname, _, filenames in os.walk(path_to_dataset):
    for filename in filenames:
        print(os.path.join(dirname, filename))



fluid_class = 9

def get_valid_idx(manualLayer):
    idx = []
    for i in range(0,61):
        temp = manualLayer[:,:,i]
        if np.sum(temp) != 0:
            idx.append(i)
    return idx



def get_valid_img_seg(mat):
    manualLayer = np.array(mat['manualLayers1'], dtype=np.uint16)
    manualFluid = np.array(mat['manualFluid1'], dtype=np.uint16)
    img = np.array(mat['images'], dtype=np.uint8)
    valid_idx = get_valid_idx(manualLayer)


    manualFluid = manualFluid[:, :, valid_idx]
    manualLayer = manualLayer[:, :, valid_idx]

    print("Seg shape", manualLayer.shape)

    seg = np.zeros((496, 768, 11))
    seg[manualFluid > 0] = fluid_class
    #cv2.imshow("d", manualFluid.astype(np.uint8)[:,:,0])
    #cv2.waitKey(100)
    max_col = -100
    min_col = 900
    for b_scan_idx in range(0, 11):
        # plt.imshow(manualFluid.astype(np.uint8)[:, :, b_scan_idx])
        # plt.show()
        for col in range(768):
            cur_col = manualLayer[:, col, b_scan_idx]
            if np.sum(cur_col) == 0:
                continue

            max_col = max(max_col, col)
            min_col = min(min_col, col)

            labels_idx = cur_col.tolist()
    #         print(f'{b_scan_idx} {labels_idx}')
    #         labels_idx.append(-1)
    #         labels_idx.insert(0, 0)
            last_st = None
            #Correccion: Salta error si el ed del label 0 es None (al principio de la iteracion)
            last_ed = 0
            for label, (st, ed) in enumerate(zip([0]+labels_idx, labels_idx+[-1])):
    #             print(st, ed)
                if st == 0 and ed == 0:
                    st = last_ed
                    if last_ed is None:
                        print(f"Last ed was none at {col}")
                    while(seg[st, col, b_scan_idx] == fluid_class):
                        st += 1

                    while(seg[st, col, b_scan_idx] != fluid_class):
                        seg[st, col, b_scan_idx] = label
                        st += 1
                        if st >= 496:
                            break
                    continue
                if ed == 0:
                    ed = st + 1
                    while(seg[ed, col, b_scan_idx] != fluid_class):
                        ed += 1

                if st == 0 and label != 0:
                    st = ed-1
                    while(seg[st, col, b_scan_idx] != fluid_class):
                        st -= 1
                    st += 1

                seg[st:ed, col, b_scan_idx] = label
                last_st = st
                last_ed = ed

    seg[manualFluid > 0] = fluid_class

    seg = seg[:, min_col:max_col+1]
    img = img[:, min_col:max_col+1]
    return img, seg, valid_idx, manualFluid


def close_fluid(img):
    zeros = np.where(img == 0)
    for x, y in zip(zeros[0], zeros[0]):
        try:
            if img[x+1, y] == fluid_class:
                img[x+1, y] = fluid_class
        except:
            pass
        try:
            if img[x-1, y] == fluid_class:
                img[x-1, y] = fluid_class
        except:
            pass
        try:
            if img[x, y+1] == fluid_class:
                img[x, y+1] = fluid_class
        except:
            pass
        try:
            if img[x, y-1] == fluid_class:
                img[x, y-1] = fluid_class
        except:
            pass


def is_fluid(fluid):
    if np.sum(fluid):
        return 'Y'
    else:
        return 'N'

mat_fps = glob(path.join(path_to_dataset, '*.mat'))

all_img = []
all_lab = []
d = 0
fluid_pat = 0
for idx_mat, m in enumerate(mat_fps):
    mat = loadmat(m)
    print("WILL LOAD ANOTHER")
    img, seg, indices, fluid = get_valid_img_seg(mat)
    #os.mkdir(f'/home/nonari/Documentos/tfgdata/patient_{idx_mat}', )

    f = 'N'
    n_fluids = 0
    for i in range(img.shape[2]):
        if i in indices:
            idx = indices.index(i)
            f = is_fluid(fluid[:, :, idx])
            if f == 'Y':
                n_fluids += 1
            plt.imsave(f'/home/nonari/Documentos/tfgdata/img_{idx_mat}_{i}_{f}.png', img[:,:,i], cmap=plt.cm.gray)
    for i in range(len(indices)):
        f = is_fluid(fluid[:, :, i])
        #close_fluid(seg[:,:,i])
        plt.imsave(f'/home/nonari/Documentos/tfgdata/seg_{idx_mat}_{indices[i]}_{f}.png', seg[:,:,i])
        d += 1
    if n_fluids > 0:
        fluid_pat +=1
    print("Patient", idx_mat, "fluids", n_fluids)

print("Total with fluids", fluid_pat)