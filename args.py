import argparse
import sys

def init_args(args=None):
    parser = argparse.ArgumentParser(description="Processing layers id.")
    parser.add_argument('--M1',  help='device process layers', required='True', type=int)
    parser.add_argument('--M2',  help='cloud process layers', required='True', type=int)
    parser.add_argument("--b1", default=0, type=int, help="samll run batch size")
    parser.add_argument("--b2", default=0, type=int, help="middle run batch size")
    parser.add_argument("--b3", default=0, type=int, help="high run batch size")
    parser.add_argument("--high_address", default="192.168.1.77:50055")         # default is edge
    parser.add_argument("--middle_address", default="192.168.1.70:50052")       #default is cloud
    return parser.parse_args(args)

def args_prase():
    args = init_args(sys.argv[1:])
    return args


