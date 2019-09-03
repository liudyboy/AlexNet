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

def send_raw_data(destination, X, Y, dest_name):
    print('send raw data to ', dest_name)
    connect.conn_send_raw_data(destination, X.array, Y.array)


edge_address = "192.168.1.77:50055"
cloud_address = "192.168.1.153:50052"

if __name__ == "__main__":
    batch_size = 128
    edge_batch, cloud_batch = batch_size, 0
    generations = 10
    for i in range(generations):
        trainX, trainY = utils.get_batch_data(batch_size)
        trainX = chainer.as_variable(trainX)
        trainY = chainer.as_variable(trainY.astype(np.int32))

        ts1 = time.time()

        if edge_batch == batch_size:
            send_raw_data(edge_address, trainX, trainY, 'edge')
        elif cloud_batch == batch_size:
            send_raw_data(cloud_address, trainX, trainY, 'cloud')

        ts2 = time.time()
        one_epoch_time = (ts2 - ts1) * 1000.
        print('one epoch cost time: ', one_epoch_time)
