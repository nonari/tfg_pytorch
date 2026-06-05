from testing import config
import sys
from os import path


layers = ["HV", "NFL", "GCL-IPL", "INL", "OPL", "OPL-ISM", "ISE", "OS-RPE", "XX", "Fluid"]

def print_header():
    print("          Loss      Dice      Recall    Specif    Accuracy  Jaccard")


def print_confusion_header():
    print("Confusion matrix")
    print("            HV      NFL   GCL-IPL   INL     OPL   OPL-ISM   ISE    OS-RPE   XX      Fluid")


def print_patient(n):
    print(f"Patient {n}")


def print_data_line(class_n, data, padding=10, skip=False):
    clss = f"{layers[class_n]}: "

    print(clss.ljust(10), end="")
    for i in data:
        print(("%.4f" % i).ljust(padding), end="")
    print()
    if skip:
        print()


def plot_table(patient, data, std=None):
    f1 = data["f1"]
    recall = data["recall"]
    specif = data["specificity"]
    accur = data["accuracy"]
    jaccard = data["jaccard"]
    loss = data["loss"]
    confusion = data["confusion"]

    sys.stdout = open(path.join(config.save_data_dir, "results.txt"), 'a')
    print_patient(patient)
    print_header()
    for i in range(10):
        print_data_line(i, [loss[i], f1[i], recall[i], specif[i], accur[i], jaccard[i]])

    if std is not None:
        f1_std = std["f1"]
        recall_std = std["recall"]
        specif_std = std["specificity"]
        accur_std = std["accuracy"]
        jaccard_std = std["jaccard"]
        loss_std = std["loss"]
        confusion_std = std["confusion"]
        print_header()
        for i in range(10):
            print_data_line(i, [loss_std[i], f1_std[i], recall_std[i], specif_std[i], accur_std[i], jaccard_std[i]])

        print()
        print_confusion_header()
        for i in range(10):
            print_data_line(i, confusion_std[:, i], padding=8)

    print()
    print_confusion_header()
    for i in range(10):
        print_data_line(i, confusion[:, i], padding=8)

    sys.stdout.close()
