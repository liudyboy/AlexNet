import argparse
import sys

def init_args(args=None):
    parser = argparse.ArgumentParser(description="Processing layers id.")
    parser.add_argument('--layers', nargs='+', type=int)
    return parser.parse_args()

def args_prase():
    args = init_args(sys.argv[1:])
    return args.layers
