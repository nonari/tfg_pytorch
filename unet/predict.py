import glob
import numpy as np
import torch
import os
import cv2
import torch.nn as nn
from unet.model import UNet
from matplotlib import pyplot
from unet.loader import ISBI_Loader, Test_Loader
from unet.loader import im_to_tensor, im_trans

if __name__ == "__main__":
    data_path = "/home/nonari/Documentos/tfgdata/tfgoct"
    isbi_dataset = Test_Loader(data_path, 0)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1,
                                               shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    # Select the device, if there is cuda use cuda, if not, use cpu
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    # Load the network, the picture is single channel, classified as 1.
    net = UNet(n_channels=1, n_classes=9)
    # Copy the network to the deivce
    net.to(device=device)
    # Load model parameters
    net.load_state_dict(torch.load('/home/nonari/Descargas/best_model_v1.pth', map_location=device))
    # Test mode
    net.eval()

    img = cv2.imread("/home/nonari/Documentos/tfgdata/tfgoct/img_9_20.png")
    lab = cv2.imread("/home/nonari/Documentos/tfgdata/tfgoct/seg_9_20.png")
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
    # Convert to batch as 1, channel as 1, size 512*512 array

    # Convert to tensor
    lab_tensor = im_to_tensor(lab)
    lab_tensor = torch.unsqueeze(lab_tensor, 0)
    img_tensor = im_trans(img)

    img_tensor = torch.unsqueeze(img_tensor, 0)
    # Copy tensor to device, only use cpu means copy to cpu, use cuda means copy to cuda.
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    lab_tensor = lab_tensor.to(device=device, dtype=torch.float32)
    # Forecast
    pred = net(img_tensor)
    # Extract result
    loss = criterion(pred, lab_tensor)
    print("Loss:", loss.item())
    pred = np.array(pred.data.cpu()[0])
    # process result
    g = pred.copy()
    g[g > 0.5] = 255
    r = np.zeros(g.shape[1:])
    for idx in range(0, 9):
        nmask = np.where(g[idx, :, :] == 255)
        r[nmask] = idx + 1
    pyplot.imshow(r)
    pyplot.show()