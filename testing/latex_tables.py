from testing import config
import sys
from os import path

table="\\begin{{table}}[]\n"\
      "\\begin{{tabular}}{{l|l|l|l|l|l|l|l|l|l|l|l|l|}}\n"\
      "\cline{{2-13}}\n"\
      "& \multicolumn{{2}}{{c|}}{{Loss}} & \multicolumn{{2}}{{c|}}{{Dice}} & \multicolumn{{2}}{{c|}}{{Sensibilidad}} & \multicolumn{{2}}{{c|}}{{Especificidad}} & \multicolumn{{2}}{{c|}}{{Precisión}} & \multicolumn{{2}}{{c|}}{{Jaccard}} \\\\ \cline{{2-13}} \n"\
      "& media        & desv       & media        & desv       & media            & desv           & media            & desv            & media          & desv          & media         & desv         \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{HV}}      & {}            & {}          & {}            & {}          & {}                & {}              & {}                & {}               & {}              & {}             & {}             & {}            \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{NFL}}     & {}            & {}          & {}            & {}          & {}                & {}              & {}                & {}               & {}              & {}             & {}             & {}            \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{GCL-IPL}} & {}            & {}          & {}            & {}          & {}                & {}              & {}                & {}               & {}              & {}             & {}             & {}            \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{INL}}     & {}            & {}          & {}            & {}          & {}                & {}              & {}                & {}               & {}              & {}             & {}             & {}            \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{OPL}}     & {}            & {}          & {}            & {}          & {}                & {}              & {}                & {}               & {}              & {}             & {}             & {}            \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{OPL-ISM}} & {}            & {}          & {}            & {}          & {}                & {}              & {}                & {}               & {}              & {}             & {}             & {}            \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{ISE}}     & {}            & {}          & {}            & {}          & {}                & {}              & {}                & {}               & {}              & {}             & {}             & {}            \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{OS-RPE}}  & {}            & {}          & {}            & {}          & {}                & {}              & {}                & {}               & {}              & {}             & {}             & {}            \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{C}}      & {}            & {}          & {}            & {}          & {}                & {}              & {}                & {}               & {}              & {}             & {}             & {}            \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{Fluid}}   & {}            & {}          & {}            & {}          & {}                & {}              & {}                & {}               & {}              & {}             & {}             & {}            \\\\ \hline\n"\
      "\end{{tabular}}\n"\
      "\end{{table}}\n"

conf_table="\\begin{{table}}[]\n" \
           "\\begin{{tabular}}{{l|l|l|l|l|l|l|l|l|l|l|}}\n" \
           "\cline{{2-11}}\n" \
           "                              & HV & NFL & GCL-IPL & INL & OPL & OPL-ISM & ISE & OS-RPE & C & Fluid \\\\ \hline\n" \
           "\multicolumn{{1}}{{|l|}}{{HV}}      & \\nicefrac{}{}   & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}      & \\nicefrac{}{} & \\nicefrac{}{}     \\\\ \hline\n" \
           "\multicolumn{{1}}{{|l|}}{{NFL}}     & \\nicefrac{}{}  & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}      & \\nicefrac{}{} & \\nicefrac{}{}     \\\\ \hline\n" \
           "\multicolumn{{1}}{{|l|}}{{GCL-IPL}} & \\nicefrac{}{}  & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}      & \\nicefrac{}{} & \\nicefrac{}{}     \\\\ \hline\n" \
           "\multicolumn{{1}}{{|l|}}{{INL}}     & \\nicefrac{}{}  & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}      & \\nicefrac{}{} & \\nicefrac{}{}     \\\\ \hline\n" \
           "\multicolumn{{1}}{{|l|}}{{OPL}}     & \\nicefrac{}{}  & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}      & \\nicefrac{}{} & \\nicefrac{}{}     \\\\ \hline\n" \
           "\multicolumn{{1}}{{|l|}}{{OPL-ISM}} & \\nicefrac{}{}  & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}      & \\nicefrac{}{} & \\nicefrac{}{}     \\\\ \hline\n" \
           "\multicolumn{{1}}{{|l|}}{{ISE}}     & \\nicefrac{}{}  & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}      & \\nicefrac{}{} & \\nicefrac{}{}     \\\\ \hline\n" \
           "\multicolumn{{1}}{{|l|}}{{OS-RPE}}  & \\nicefrac{}{}  & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}      & \\nicefrac{}{} & \\nicefrac{}{}     \\\\ \hline\n" \
           "\multicolumn{{1}}{{|l|}}{{C}}       & \\nicefrac{}{}  & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}      & \\nicefrac{}{} & \\nicefrac{}{}     \\\\ \hline\n" \
           "\multicolumn{{1}}{{|l|}}{{Fluid}}   & \\nicefrac{}{}  & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}   & \\nicefrac{}{}       & \\nicefrac{}{}   & \\nicefrac{}{}      & \\nicefrac{}{} & \\nicefrac{}{}     \\\\ \hline\n" \
           "\end{{tabular}}\n" \
           "\end{{table}}\n"

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


def plot_table(patient, data, std):
    f1 = data["f1"]
    recall = data["recall"]
    specif = data["specificity"]
    accur = data["accuracy"]
    jaccard = data["jaccard"]
    loss = data["loss"]
    confusion = data["confusion"]

    sys.stdout = open(path.join(config.save_data_dir, "latex.txt"), 'a')

    f1_std = std["f1"]
    recall_std = std["recall"]
    specif_std = std["specificity"]
    accur_std = std["accuracy"]
    jaccard_std = std["jaccard"]
    loss_std = std["loss"]
    confusion_std = std["confusion"]
    data_fomat = []
    for i in range(10):
        data_fomat += [loss[i], loss_std[i], f1[i], f1_std[i], recall[i], recall_std[i], specif[i], specif_std[i],
                       accur[i], accur_std[i], jaccard[i], jaccard_std[i]]

    data_fomat = list(map(lambda n: trunc(n), data_fomat))
    print(table.format(*data_fomat))

    print()
    data_fomat.clear()
    for i in range(10):
        std_c = confusion_std[:, i].tolist()
        conf = confusion[:, i]
        for a, b in zip(std_c, conf):
            data_fomat += ['{'+trunc(b)+'}', '{'+trunc(a)+'}']
    print(conf_table.format(*data_fomat))

    sys.stdout.close()


def trunc(num):
    s = str(num)
    if len(s) > 5:
        if float(s[:5]) == 0:
            return '0.0'
        return s[:5]
    else:
        return s
