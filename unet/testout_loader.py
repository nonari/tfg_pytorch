import torch
import cv2
import os
import glob
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import random
import numpy as np
import ntpath
from utils import t_utils
from PIL import Image


def im_to_tensor(im):
    d = np.unique(im)

    l = []
    for idx, n in enumerate(d):
        i = np.where(im == n)
        t = np.zeros((im.shape[0], im.shape[1]), np.float32)
        t[i] = 1
        t = transforms.ToPILImage()(t)
        t = transforms.Resize((512, 512), transforms.InterpolationMode.NEAREST)(t)
        l.append(t)
    if len(d) == 9:
        t = np.zeros((im.shape[0], im.shape[1]), np.float32)
        t = transforms.ToPILImage()(t)
        t = transforms.Resize((512, 512), transforms.InterpolationMode.NEAREST)(t)
        l.append(t)
    st = np.dstack(tuple(l))
    st = np.swapaxes(st, 0, 2)
    st = np.swapaxes(st, 1, 2)
    tensor = torch.from_numpy(st)
    return tensor


im_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Resize((512, 512), transforms.InterpolationMode.BILINEAR)
])


class Loader(Dataset):
    def __init__(self, data_path):
        # Initialization function, read all the pictures under data_path
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, f'img_*'))

    def __getitem__(self, index):
        # Read pictures according to index
        image_path = self.imgs_path[index]
        # Read training pictures and label pictures
        image = cv2.imread(image_path)
        # Convert the data to a single channel picture
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = im_trans(image)
        # Process the label, change the pixel value of 255 to 1
        return image, {}

    def __len__(self):
        # Return the training set size
        return len(self.imgs_path)


class Test_Loader(Loader):
    def __init__(self, data_path):
        # Initialization function, read all the pictures under data_path
        super().__init__(data_path)

    def __getitem__(self, index):
        img, seg = Loader.__getitem__(self, index)
        image_path = self.imgs_path[index]
        filename = ntpath.basename(image_path)
        return img, seg, filename
