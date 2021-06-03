import torch
import cv2
import os
import glob
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import random
import numpy as np



def im_to_tensor(im):
    d = np.unique(im)

    l = []
    for idx, n in enumerate(d[1:]):
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

class ISBI_Loader(Dataset):
    def __init__(self, data_path, patient_left=0):
        # Initialization function, read all the pictures under data_path
        self.data_path = data_path
        self.patient_left = patient_left
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

        image = im_trans(image)
        # Process the label, change the pixel value of 255 to 1
        return image, im_to_tensor(label)

    def __len__(self):
        # Return the training set size
        return len(self.imgs_path)


class Test_Loader(ISBI_Loader):
    def __init__(self, data_path, patient_left=0):
        # Initialization function, read all the pictures under data_path
        super().__init__(data_path, patient_left)
        self.imgs_path = glob.glob(os.path.join(data_path, f'img_{patient_left}_*'))


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("data/train/")
    print("Number of data:", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                           batch_size=2,
                                           shuffle=True)
    for image, label in train_loader:
        print(image.shape)