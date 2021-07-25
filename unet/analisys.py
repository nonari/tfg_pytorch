import glob
import os
import ast
from statistics import stdev
from matplotlib import pyplot
import numpy as np
data_dir = "/home/nonari/Documents/resnet34_cp_sa"
data_path_acc = f"{data_dir}/accuracy"
data_path_loss = f"{data_dir}/loss"


def avg(arr):
    l = []
    for i in zip(*tuple(arr)):
        avg = sum(i) / len(i)
        l.append(avg)
    return np.array(l)


def dev(arr):
    l = []
    for i in zip(*tuple(arr)):
        std = stdev(i)
        l.append(std)
    return np.array(l)


def extract(test_files):
    loss_arr_train_all = []
    epoch_arr_train_all = []
    for f in test_files:
        with open(f, 'r') as file:
            lines = file.readlines()
            parsed = [ast.literal_eval(n) for n in lines]
            loss_arr_train = [i for i, j in parsed]
            epoch_arr_train = [j for i, j in parsed]
            last_epoch = -1
            loss_arr_train_filter = []
            epoch_arr_train_filter = []
            for loss, epoch in zip(loss_arr_train, epoch_arr_train):
                if epoch != last_epoch:
                    last_epoch = epoch
                    loss_arr_train_filter.append(loss)
                    epoch_arr_train_filter.append(epoch)
        loss_arr_train_all.append(loss_arr_train_filter)
        epoch_arr_train_all.append(epoch_arr_train_filter)
    return loss_arr_train_all, epoch_arr_train_all


test_files_acc = glob.glob(os.path.join(data_path_acc, f'train_*'))
test_files_loss = glob.glob(os.path.join(data_path_loss, f'train_*'))

acc_train_all, epoch_acc_train_all = extract(test_files_acc)
loss_train_all, epoch_acc_train_all = extract(test_files_loss)

# pyplot.plot(epoch_arr_train_all[0], loss_arr_train_all[0])
# pyplot.title("train 0")
# pyplot.show()
#
# pyplot.plot(epoch_arr_train_all[1], loss_arr_train_all[1])
# pyplot.title("train 1")
# pyplot.show()

acc = avg(acc_train_all)
acc_stdev = dev(acc_train_all)
loss = avg(loss_train_all)
loss_stdev = dev(loss_train_all)

pyplot.plot(epoch_acc_train_all[0], acc)
pyplot.fill_between(epoch_acc_train_all[0], acc-acc_stdev, acc + acc_stdev, alpha=0.5)
pyplot.title("NP AVG accuracy")
pyplot.show()

pyplot.plot(epoch_acc_train_all[0], loss)
pyplot.fill_between(epoch_acc_train_all[0], loss-loss_stdev, loss + loss_stdev, alpha=0.5)
pyplot.title("NP AVG Loss")
pyplot.show()