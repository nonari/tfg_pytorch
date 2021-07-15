from testing.train import tt
import argparse
from utils.config import set_data


def args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('traindata', type=str)
    argument_parser.add_argument('savedata', type=str)
    argument_parser.add_argument('encoder', type=str)
    argument_parser.add_argument('--ini', type=int, default=0)
    argument_parser.add_argument('--end', type=int, default=10)
    argument_parser.add_argument('--lr', type=float, default=0.00001)
    argument_parser.add_argument('--epochs', type=float, default=175)
    argument_parser.add_argument('--weights', type=str, default=None)
    argument_parser.add_argument('--augment', action="store_true")
    parsed_args = argument_parser.parse_args()

    print(parsed_args)
    set_data(parsed_args)


if __name__ == '__main__':
    args()
    tt()
