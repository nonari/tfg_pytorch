# This is a sample Python script.
from testing.train import tt
import argparse
from utils.config import set_data
# Press Alt+Mayús+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('traindata', type=str)
    argument_parser.add_argument('savedata', type=str)
    argument_parser.add_argument('encoder', type=str)
    argument_parser.add_argument('--use-imagenet', action="store_true")
    argument_parser.add_argument('--augment', action="store_true")
    args = argument_parser.parse_args()
    argobj = {"train_data_dir": args.traindata,
              "save_data_dir": args.savedata,
              "encoder": args.encoder,
              "use_imagenet": args.use_imagenet,
              "augment": args.augment}
    print(args)
    set_data(argobj)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+Mayús+B to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args()
    tt()
