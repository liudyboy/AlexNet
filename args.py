import argparse
import sys

def init_args(args=None):
    parser = argparse.ArgumentParser(description="Processing layers id.")
    parser.add_argument('-M1', '--device', help='device process layers', required='True', type=int)
    parser.add_argument('-M2', '--cloud', help='cloud process layers', required='True', type=int)
    return parser.parse_args(args)

def args_prase():
    args = init_args(sys.argv[1:])
    return (args.device, args.cloud)


