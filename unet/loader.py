import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # Initialization function, read all the pictures under data_path
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'img*'))

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
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # Process the label, change the pixel value of 255 to 1
        return image, label

    def __len__(self):
        # Return the training set size
        return len(self.imgs_path)


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("data/train/")
    print("Number of data:", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                           batch_size=2,
                                           shuffle=True)
    for image, label in train_loader:
        print(image.shape)