import torch
import cv2
import os
import glob
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import torch
import random
import numpy as np
import ntpath
from utils import t_utils
from PIL import Image


def im_to_tensor(im):
    d = np.unique(im).tolist()
    l = []
    for idx, n in enumerate(d):
        i = np.where(im == n)
        t = np.zeros((im.shape[0], im.shape[1]), np.float32)
        t[i] = 1
        t = transforms.ToPILImage()(t)
        t = transforms.Resize((512, 512), Image.NEAREST)(t)
        l.append(t)
    if len(d) == 9:
        t = np.zeros((im.shape[0], im.shape[1]), np.float32)
        t = transforms.ToPILImage()(t)
        t = transforms.Resize((512, 512), Image.NEAREST)(t)
        l.append(t)
    st = np.dstack(tuple(l))
    st = np.swapaxes(st, 0, 2)
    st = np.swapaxes(st, 1, 2)
    tensor = torch.from_numpy(st)
    return tensor


im_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Resize((512, 512), Image.BILINEAR)
])


def augment(im, mask):
    im = transforms.ToTensor()(im)
    i, j, h, w = transforms.RandomResizedCrop.get_params(im, [0.95, 1], [1, 1])
    im = F.resized_crop(im, i, j, h, w, [512, 512], interpolation=Image.BILINEAR)
    mask = F.resized_crop(mask, i, j, h, w, [512, 512], interpolation=Image.NEAREST)

    ang = transforms.RandomRotation.get_params([-5, 5])
    im = F.rotate(im, ang, interpolation=Image.BILINEAR)
    mask = F.rotate(mask, ang, interpolation=Image.NEAREST)

    if torch.rand(1) < 0.4:
        im = F.hflip(im)
        mask = F.hflip(mask)

    return torch.squeeze(im).numpy(), mask


class ISBI_Loader(Dataset):
    def __init__(self, data_path, patient_left=0, augment=False):
        # Initialization function, read all the pictures under data_path
        self.data_path = data_path
        self.patient_left = patient_left
        self.augment = augment
        discard = glob.glob(os.path.join(data_path, f'img_{patient_left}_*'))
        self.imgs_path = glob.glob(os.path.join(data_path, f'img_*'))
        for path in discard:
            self.imgs_path.remove(path)

    def __getitem__(self, index):
        # Read pictures according to index
        image_path = self.imgs_path[index]
        # Generate label_path according to image_path
        label_path = image_path.replace('img', 'seg')
        # Read training pictures and label pictures
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # Convert the data to a single channel picture
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        label = im_to_tensor(label)

        if self.augment and random.random() > 0.5:
            image, label = augment(image, label)

        image = im_trans(image)

        return image, label

    def __len__(self):
        # Return the training set size
        return len(self.imgs_path)


class Test_Loader(ISBI_Loader):
    def __init__(self, data_path, patient_left=0):
        # Initialization function, read all the pictures under data_path
        super().__init__(data_path, patient_left)
        self.imgs_path = glob.glob(os.path.join(data_path, f'img_{patient_left}_*'))

    def __getitem__(self, index):
        img, seg = ISBI_Loader.__getitem__(self, index)
        image_path = self.imgs_path[index]
        filename = ntpath.basename(image_path)
        return img, seg, filename


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("data/train/")
    print("Number of data:", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                           batch_size=2,
                                           shuffle=True)
    for image, label in train_loader:
        print(image.shape)