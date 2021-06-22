import torch
import torch.nn as nn
from unet.testout_loader import Test_Loader
import segmentation_models_pytorch as smp
from sklearn import metrics
from utils import t_utils
import glob
import os
from utils import m_utils
import numpy as np
import matplotlib.pyplot as plt
from utils.f_utils import save_obj
from testing.plot_tables import plot_table
from testing.plot_tables_no_back import plot_table as plot_no_back
import re

data_path = "/home/nonari/Documentos/tfgdata/tfgoct/"
models_path = "/home/nonari/Descargas/models_10c/"
info_data = ""


def test(net, img_tensor):
    net.eval()

    pred = net(img_tensor)

    mask = t_utils.prediction_to_mask_x(pred)
    return mask


if __name__ == "__main__":
    test_results_acc = []

    models = glob.glob(os.path.join(models_path, f'best_model_p*'))
    models = sorted(models)
    model = models[0]
    isbi_dataset = Test_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1,
                                               shuffle=False)
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    net = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pretreined weights for encoder initialization
        in_channels=1,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=10,  # model output channels (number of classes in your dataset)
    )
    net.load_state_dict(torch.load(model, map_location=device))

    for img_tensor, _, filename in train_loader:
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        mask = test(net, img_tensor)
        plt.imsave(f"/home/nonari/Documentos/tfgdata/test_result_mask/{filename}", mask)

