from unet.loader import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from utils.f_utils import save_line
import segmentation_models_pytorch as smp
from utils import t_utils, m_utils
from sklearn import metrics
import numpy as np

data_path = "/home/nonari/Documentos/tfgdata/tfgoct/"
loss_data_path = "/mydrive/FIC/TFG/train_unet_final/no_pretrained/loss/"
accuracy_data_path = "/mydrive/FIC/TFG/train_unet_final/no_pretrained/accuracy/"
model_data_path = "/mydrive/FIC/TFG/train_unet_final/no_pretrained/models/"



def split_acc(tensor_true, tensor_pred):
    all_l = []
    for i in range(0, tensor_true.shape[0]):
        l = m_utils.accuracy(tensor_true[i].flatten()+1, tensor_pred[i].flatten()+1)
        all_l.append(l)
    res = []
    for t in zip(*tuple(all_l)):
        avg_l = sum(t) / len(t)
        res.append(avg_l)

    return res


def train_net(net, device, isbi_dataset, epochs=175, batch_size=9, lr=0.00001):
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')

    for epoch in range(epochs):
        net.train()

        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            pred = net(image)
            mask_pred = np.argmax(pred.data.cpu().numpy(), axis=1)
            mask_label = t_utils.tensor_to_ml_mask(label)
            void_pos = np.where(mask_label == -1)
            mask_pred[void_pos] = -1
            label = label.to(device=device, dtype=torch.float32)
            accuracy = metrics.accuracy_score(mask_pred.flatten(), mask_label.flatten(), normalize=True)
            #accuracy_layers = split_acc(mask_pred, mask_label)
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            print('Accuracy', accuracy)
            save_line((loss.item(), epoch), f"{loss_data_path}train_unet_p{isbi_dataset.patient_left}.txt")
            save_line((accuracy, epoch), f"{accuracy_data_path}train_unet_p{isbi_dataset.patient_left}.txt")
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), f'{model_data_path}best_model_p{isbi_dataset.patient_left}.pth')

            loss.backward()
            optimizer.step()


def tt():
    for i in range(0, 10):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=10,
        )

        net.to(device=device)

        isbi_dataset = ISBI_Loader(data_path, i, augment=False)
        train_net(net, device, isbi_dataset)
        del device
        del net
        torch.cuda.empty_cache()
