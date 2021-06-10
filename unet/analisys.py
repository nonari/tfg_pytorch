import glob
import os
import ast
from matplotlib import pyplot
data_path = "/home/nonari/Documentos/tfgdata/"

test_files = glob.glob(os.path.join(data_path, f'test_*'))

loss_arr_test_all = []
epoch_arr_test_all = []
for f in test_files:
    with open(f, 'r') as file:
        lines = file.readlines()
        parsed = [ast.literal_eval(n) for n in lines]
        loss_arr_test = [i for i, j in parsed]
        epoch_arr_test = [j for i, j in parsed]
    loss_arr_test_all.append(loss_arr_test)
    epoch_arr_test_all.append(epoch_arr_test)
test_files = glob.glob(os.path.join(data_path, f'train_*'))

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

pyplot.plot(epoch_arr_train_all[0], loss_arr_train_all[0])
pyplot.title("train 0")
pyplot.show()

pyplot.plot(epoch_arr_train_all[1], loss_arr_train_all[1])
pyplot.title("train 1")
pyplot.show()

pyplot.plot(epoch_arr_test_all[0], loss_arr_test_all[0])
pyplot.title("test 0")
pyplot.show()

pyplot.plot(epoch_arr_test_all[1], loss_arr_test_all[1])
pyplot.title("test 1")
pyplot.show()

