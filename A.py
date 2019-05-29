from __future__ import print_function
import logging
import numpy as np

import grpc
import communication_pb2
import communication_pb2_grpc
import pickle
import chainer
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import link
from chainer import initializers
from chainer import utils
from chainer import variable
import update
import layersM as layers
import tiny_utils as utils
import gc
import time
import sys
import args
import _thread
import connect
# from threading import Thread

edge_address = "192.168.1.77:50055"
cloud_address = "192.168.1.70:50055"

def wait_threads_complete():
    global finish_flag 
    target = 3
    while finish_flag is not target:
        pass
    print('one epoch training completed')
def send_raw_data(threadName, threadID, destination, X, Y):
    ts1 = time.time()
    global finish_flag
    print(threadName, " start time: ", ts1)
    singal = connect.conn_send_raw_data(destination, X.array, Y.array)
    print("singal shape: ", singal.shape)
    ts2 = time.time()
    print(threadName, " end time: ", ts2)
    print(threadName, " cost time: ", (ts2-ts1)*1000)
    print(threadName, " end")
    finish_flag += 1

def send_output_data(destination, output, Y):
    global finish_flag
    print('Send output to cloud')
    ts1 = time.time()
    reply = connect.conn_send_device_output_data(destination, output.array, Y.array)
    ts2 = time.time()
    print('send output to cloud cost time: ', (ts2 - ts1) * 1000.)
    finish_flag += 1

if __name__ == "__main__":
    # array = np.arange(2000*1000)
    # np.savez("data", x=array)
    finish_flag = 0
    # cloud_x = chainer.as_variable(np.arange(1000*1000))
    cloud_x = chainer.as_variable(np.arange(1))
    cloud_Y = chainer.as_variable(np.array([1]))
    ts1 = time.time()
    try:
        _thread.start_new_thread(send_raw_data, ("send cloud thread", 0, cloud_address, cloud_x, cloud_Y))
        # _thread.start_new_thread(send_raw_data, ("send edge thread", 1, edge_address, cloud_x, cloud_Y))
    except:
        print('send raw thread error')
    cloud_x = chainer.as_variable(np.arange(1))
    # cloud_x = chainer.as_variable(np.arange(2000*1000))

    send_output_data(cloud_address, cloud_x, cloud_Y)
    wait_threads_complete()
    ts2 = time.time()
    print("total cost time: ", (ts2 - ts1) * 1000)
    while True:
        pass
