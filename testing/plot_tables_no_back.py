
def print_header():
    print("          Dice      Recall    Jaccard")


def print_confusion_header():
    print("Confusion matrix")
    print("            0       1       2       3       4       5       6       7       8       Fluid")


def print_patient(n, loss):
    print(f"Patient {n}")
    print(f"Loss: {loss}")


def print_data_line(class_n, data, padding=10, skip=False):
    clss = f"Layer {class_n}: "
    if class_n == 9:
        clss = "Fluid:"
    print(clss.ljust(10), end="")
    for i in data:
        print(("%.4f" % i).ljust(padding), end="")
    print()
    if skip:
        print()


def plot_table(patient, data):
    f1 = data["f1"]
    recall = data["recall"]
    jaccard = data["jaccard"]
    loss = data["loss"]
    confusion = data["confusion"]


    print_patient(patient, loss)
    print_header()
    for i in range(10):
        print_data_line(i, [f1[i], recall[i], jaccard[i]])

    print()
    print_confusion_header()
    for i in range(10):
        print_data_line(i, confusion[:, i], padding=8)