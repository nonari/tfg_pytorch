from testing.metrics_l import ts
import argparse
from testing.config import set_data


def args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('modelsdir', type=str)
    argument_parser.add_argument('testdir', type=str)
    argument_parser.add_argument('savedir', type=str)
    argument_parser.add_argument('encoder', type=str)
    parsed_args = argument_parser.parse_args()

    print(parsed_args)
    set_data(parsed_args)


if __name__ == '__main__':
    args()
    ts()
