import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import cv2


def prediction_to_mask(tensor):
    tn = tensor.clone().detach()
    # TODO this method leaves background gaps
    tn[tn > 0.5] = 255
    mask = np.zeros(tn.shape[2:])
    for idx in range(0, 10):
        layer = np.where(tn[0, idx, :, :] == 255)
        mask[layer] = idx + 1
    return mask


def prediction_to_mask_x(tensor):
    tn = tensor.clone().detach()
    arr = torch.squeeze(tn, dim=0).numpy()
    mask = np.argmax(arr, axis=0)
    mask = mask + 1

    return mask


mask_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512), Image.NEAREST),
    transforms.PILToTensor()
])


def mask_to_tensor(mask):
    d = np.unique(mask)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    tensor = np.zeros((10, mask.shape[0], mask.shape[1]), np.float32)
    for idx, n in enumerate(d):
        x, y = np.where(mask == n)
        tensor[idx, x, y] = 1

    tensor = torch.from_numpy(tensor)

    return tensor


def tensor_to_mask(tensor):
    tensor = torch.squeeze(tensor, 0)
    tensor = tensor.transpose(0, 2)
    tensor = tensor.transpose(0, 1)
    v = np.arange(tensor.shape[2]) + 1
    tensor = tensor * v
    mask = torch.sum(tensor, 2)

    return mask.numpy()


def tensor_to_ml_mask(tensor):
    v = np.arange(tensor.shape[1]) + 1

    tensor = tensor * v.reshape((1, 10, 1, 1))
    mask = torch.sum(tensor, 1) - 1

    return mask.numpy()