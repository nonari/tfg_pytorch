from testing import config
import sys
from os import path
from sklearn.metrics import ConfusionMatrixDisplay
import decimal
from matplotlib import pyplot as plt

# create a new context for this task
ctx = decimal.Context()

# 20 digits should be enough for everyone :D
ctx.prec = 20

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')


table="\\begin{{table}}[]\n"\
      "\\begin{{tabular}}{{l|l|l|l|l|l|l|l|l|l|l|l|l|}}\n"\
      "\cline{{2-13}}\n"\
      "& \multicolumn{{2}}{{c|}}{{Loss}} & \multicolumn{{2}}{{c|}}{{Dice}} & \multicolumn{{2}}{{c|}}{{Sensibilidad}} & \multicolumn{{2}}{{c|}}{{Especificidad}} & \multicolumn{{2}}{{c|}}{{Precisión}} & \multicolumn{{2}}{{c|}}{{Jaccard}} \\\\ \cline{{2-13}} \n"\
      "& media & desv & media & desv & media & desv & media & desv & media & desv & media & desv \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{HV}} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{NFL}} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{GCL-IPL}} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{INL}} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{OPL}} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{OPL-ISM}} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{ISE}} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{OS-RPE}} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n"\
      "\multicolumn{{1}}{{|l|}}{{C}} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n"\
      "\end{{tabular}}\n"\
      "\end{{table}}\n"

conf_table_wstd="\\begin{{table}}[]\n" \
"\\begin{{tabular}}{{|l|l|l|l|l|l|l|l|l|l|l|}}\n" \
"\hline\n" \
"\\backslashbox{{Real}}{{Pred}} & HV & NFL & GCL-IPL & INL & OPL & OPL-ISM & ISE & OS-RPE & C & Fluid \\\\ \hline\n" \
"HV & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} \\\\ \hline\n" \
"NFL & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} \\\\ \hline\n" \
"GCL-IPL & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} \\\\ \hline\n" \
"INL & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} \\\\ \hline\n" \
"OPL & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} \\\\ \hline\n" \
"OPL-ISM & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} \\\\ \hline\n" \
"ISE & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} \\\\ \hline\n" \
"OS-RPE & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} \\\\ \hline\n" \
"C & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} & \\nicefrac{}{} \\\\ \hline\n" \
"\end{{tabular}}\n" \
"\end{{table}}\n"

conf_table="\\begin{{table}}[]\n" \
"\\begin{{tabular}}{{|l|l|l|l|l|l|l|l|l|l|l|}}\n" \
"\hline\n" \
"\\backslashbox{{Real}}{{Pred}} & HV & NFL & GCL-IPL & INL & OPL & OPL-ISM & ISE & OS-RPE & C & Fluid \\\\ \hline\n" \
"HV & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n" \
"NFL & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n" \
"GCL-IPL & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n" \
"INL & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n" \
"OPL & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n" \
"OPL-ISM & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n" \
"ISE & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n" \
"OS-RPE & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n" \
"C & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \hline\n" \
"\end{{tabular}}\n" \
"\end{{table}}\n"

layers = ["HV", "NFL", "GCL-IPL", "INL", "OPL", "OPL-ISM", "ISE", "OS-RPE", "C"]

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
    exp_name = path.basename(config.save_data_dir).__str__()
    f1 = data["f1"]
    recall = data["recall"]
    # specif = data["specificity"]
    accur = data["accuracy"]
    jaccard = data["jaccard"]
    loss = data["loss"]
    confusion = data["confusion"]

    sys.stdout = open(path.join(config.save_data_dir, f'{exp_name}.txt'), 'a')

    f1_std = std["f1"]
    recall_std = std["recall"]
    # specif_std = std["specificity"]
    accur_std = std["accuracy"]
    jaccard_std = std["jaccard"]
    loss_std = std["loss"]
    confusion_std = std["confusion"]
    data_fomat = []
    for i in range(9):
        data_fomat += [loss[i], loss_std[i], f1[i], f1_std[i], recall[i], recall_std[i], 0, 0,
                       accur[i], accur_std[i], jaccard[i], jaccard_std[i]]

    data_fomat = list(map(lambda n: trunc(n), data_fomat))
    print(table.format(*data_fomat))

    sys.stdout.close()

    confusion = (confusion * 1000).astype(int)
    confusion = confusion / 10
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=layers)
    conf_disp.plot(xticks_rotation=25, values_format='.1f')
    plt.savefig(path.join(config.save_data_dir, f'{exp_name}_conf.png'), bbox_inches="tight")

def trunc(num):
    s = float_to_str(num)
    if len(s) > 5:
        if float(s[:5]) == 0:
            return '0.0'
        return s[:5]
    else:
        return s
