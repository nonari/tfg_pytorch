# This is a sample Python script.
from testing.train import tt
import argparse
# Press Alt+Mayús+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--train-set', type=str)
    argument_parser.add_argument('--models', type=str)
    argument_parser.add_argument('--loss', type=str)
    argument_parser.add_argument('--accuracy', type=str)
    argument_parser.add_argument('--use-imagenet')
    argument_parser.add_argument('encoder', choices=["1", "2"])
    args = argument_parser.parse_args()



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+Mayús+B to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tt()
