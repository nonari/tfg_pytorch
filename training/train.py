from unet.loader import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from utils.f_utils import save_line
import segmentation_models_pytorch as smp
from utils import t_utils, m_utils, f_utils
from sklearn import metrics
import numpy as np
import training.config as config
import os


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
    optimizer.zero_grad()
    for epoch in range(epochs):
        net.train()

        acc_total = 0
        loss_total = 0
        for idx, (image, label) in enumerate(train_loader):
            image = image.to(device=device, dtype=torch.float32)
            pred = net(image)
            mask_pred = np.argmax(pred.data.cpu().numpy(), axis=1)
            mask_label = t_utils.tensor_to_ml_mask(label)
            void_pos = np.where(mask_label == -1)
            mask_pred[void_pos] = -1
            label = label.to(device=device, dtype=torch.float32)
            accuracy = metrics.accuracy_score(mask_pred.flatten(), mask_label.flatten(), normalize=True)
            acc_total += accuracy
            #accuracy_layers = split_acc(mask_pred, mask_label)
            loss = criterion(pred, label)
            loss.backward()
            loss_total += loss.item()
            if idx % config.parts == 0:
                batch_loss = loss_total / config.parts
                batch_acc = acc_total / config.parts
                print('Loss/train', batch_loss)
                print('Accuracy', batch_acc)
                data_file = f'train_unet_p{isbi_dataset.patient_left}.txt'
                save_line((batch_loss, epoch), os.path.join(config.save_data_dir, "loss", data_file))
                save_line((batch_acc, epoch), os.path.join(config.save_data_dir, "accuracy", data_file))
                if batch_loss <= best_loss:
                    best_loss = batch_loss
                    model_file = f"best_model_p{isbi_dataset.patient_left}.pth"
                    torch.save(net.state_dict(), os.path.join(config.save_data_dir, "models", model_file))
                optimizer.step()
                optimizer.zero_grad()



def tt():
    encoder = config.encoder
    pretraining = config.weights
    f_utils.create_skel(config.save_data_dir)
    for i in range(config.ini, config.end):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = smp.Unet(
            encoder_name=encoder,
            encoder_weights=pretraining,
            in_channels=1,
            classes=10,
        )

        net.to(device=device)
        isbi_dataset = ISBI_Loader(config.train_data_dir, i, augment=config.augment)
        train_net(net, device, isbi_dataset, epochs=config.epochs, lr=config.lr)
        del device
        del net
        torch.cuda.empty_cache()
