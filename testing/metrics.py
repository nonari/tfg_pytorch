import torch
import torch.nn as nn
from unet.loader import Test_Loader
import segmentation_models_pytorch as smp
from sklearn import metrics
from utils import t_utils
import glob
import os
from utils import m_utils
import numpy as np
import matplotlib.pyplot as plt
from testing.plot_tables import plot_table
import re
from os import path
from testing import config
from utils import f_utils


def test(net, img_tensor, lab_tensor):
    net.eval()

    pred = net(img_tensor)
    criterion = nn.BCEWithLogitsLoss()

    loss = criterion(pred, lab_tensor)
    #print("Test/Loss:", loss.item())

    actual_label = t_utils.tensor_to_mask(lab_tensor).flatten()
    pred_label = t_utils.prediction_to_mask(pred).flatten()

    mask = t_utils.prediction_to_mask(pred)

    jaccard = metrics.jaccard_score(actual_label, pred_label, average=None)
    recall = metrics.recall_score(actual_label, pred_label, average=None, zero_division=1)
    f1 = metrics.f1_score(actual_label, pred_label, average=None)
    confusion = metrics.confusion_matrix(actual_label, pred_label)
    if confusion.shape[0] == 10:
        confusion = np.hstack((confusion, np.zeros(10)[:, np.newaxis]))
        confusion = np.vstack((confusion, np.zeros(11)))
        confusion[10, 10] = 1
        jaccard = np.append(jaccard, 1)
        recall = np.append(recall, 1)
        f1 = np.append(f1, 1)

    total_by_class = np.sum(confusion, axis=1)
    if total_by_class[10] == 0:
        confusion[10, 10] = 1
        total_by_class[10] = 1
    total_by_class = total_by_class[:, np.newaxis]
    confusion = confusion / total_by_class

    return {"loss": loss.item(), "jaccard": jaccard, "recall": recall, "f1": f1, "confusion": confusion, "mask": mask}


def test_no_back(net, img_tensor, lab_tensor):
    net.eval()

    pred = net(img_tensor)
    criterion = nn.CrossEntropyLoss()

    # loss = criterion(pred, lab_tensor)
    #print("Test/Loss:", loss.item())
    losses = []
    for i in range(0, 10):
        #lss = criterion(pred[0, i], lab_tensor[0, i])
        losses.append(0)
    actual_label = t_utils.tensor_to_mask(lab_tensor).flatten()
    pred_label = t_utils.prediction_to_mask_x(pred).flatten()

    mask = t_utils.prediction_to_mask_x(pred)

    jaccard = metrics.jaccard_score(actual_label, pred_label, average=None)
    recall = metrics.recall_score(actual_label, pred_label, average=None, zero_division=1)
    f1 = metrics.f1_score(actual_label, pred_label, average=None)
    specificity = m_utils.specificity(actual_label, pred_label, 10)
    confusion = metrics.confusion_matrix(actual_label, pred_label)
    if confusion.shape[0] == 9:
        confusion = np.hstack((confusion, np.zeros(9)[:, np.newaxis]))
        confusion = np.vstack((confusion, np.zeros(10)))
        confusion[9, 9] = 1
        jaccard = np.append(jaccard, 1)
        recall = np.append(recall, 1)
        f1 = np.append(f1, 1)

    total_by_class = np.sum(confusion, axis=1)
    if total_by_class[9] == 0:
        confusion[9, 9] = 1
        total_by_class[9] = 1
    total_by_class = total_by_class[:, np.newaxis]
    confusion = confusion / total_by_class
    accuracy = m_utils.accuracy(actual_label, pred_label)

    return {"loss": losses, "jaccard": jaccard, "recall": recall, "specificity": specificity, "accuracy": accuracy,
            "f1": f1, "confusion": confusion, "mask": mask}


def ts():
    test_results_acc = []
    f_utils.create_dir(config.save_data_dir)
    models = glob.glob(os.path.join(config.models_dir, f'best_model_p*'))
    models = sorted(models)

    patients_avgs = []
    for idx, model in enumerate(models):
        m = re.search('best_model_p(\d+)\.pth', model)
        idx = m.group(1)
        isbi_dataset = Test_Loader(config.test_images_dir, idx)
        train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                                   batch_size=1,
                                                   shuffle=False)
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        net = smp.Unet(
            encoder_name=config.encoder,
            encoder_weights=None,
            in_channels=1,
            classes=10,
        )
        net.load_state_dict(torch.load(model, map_location=device))

        patient_results = []
        for img_tensor, lab_tensor, filename in train_loader:
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            lab_tensor = lab_tensor.to(device=device, dtype=torch.float32)
            results = test_no_back(net, img_tensor, lab_tensor)
            patient_results.append(results)
            mask = results["mask"]
            filename = filename[0].replace("img", "seg")
            mask_path = path.join(config.save_data_dir, filename)
            plt.imsave(mask_path, mask)

        avg_patient_results = m_utils.average_metrics(patient_results)
        patients_avgs.append(avg_patient_results)
        plot_table(idx, avg_patient_results)
        del net
        del device

    all_avg = m_utils.average_metrics(patients_avgs)
    all_std = m_utils.stdev_metrics(patients_avgs)
    plot_table("ALL", all_avg, std=all_std)
