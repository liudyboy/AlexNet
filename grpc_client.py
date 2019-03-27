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
import layers
import utils
import gc
import time
import sys





def run(sendArray, Y):
    with grpc.insecure_channel("192.168.1.153:50051", options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)]) as channel:
        stub = communication_pb2_grpc.CommStub(channel)
        sendArray = pickle.dumps(sendArray)
        Y = pickle.dumps(Y)
        recv_array = stub.Forwarding(communication_pb2.ArrayRecv(array=sendArray, Y=Y))
        recv_array = pickle.loads(recv_array.array)
    return recv_array

if __name__ == "__main__":
    logging.basicConfig()
    ts = time.time()
    start_time = time.ctime(ts)
    print("start time:", start_time)
    generations = 1000
    batch_size = 128


    for i in range(generations):

        ts1 = time.time()

        trainX, trainY = utils.get_batch_data(batch_size)

        Y = trainY.astype(np.int32)
        Y = chainer.as_variable(Y)
        trainX = chainer.as_variable(trainX)

        output = trainX

        ts3 = time.time()
        dout = run(output, Y)

        ts4 = time.time()

        ts2 = time.time()
        process_time = ts2 - ts1
        client_compute_time = ts3-ts1 + ts2-ts4

        if i is not 0:
            if i == 1:
                complete_time = process_time*1000.
                client_used_time = client_compute_time*1000.
                server_used_time = (complete_time - client_used_time)
            elif i > 1:
                complete_time = (process_time*1000.)/i + complete_time*(i-1)/i
                client_used_time = (client_compute_time*1000.)/i + client_used_time*(i-1)/i
                server_used_time = complete_time - client_used_time

            print("#epoch {} completed!  Used time {}".format(i, complete_time))
            print("client computing time: ", client_used_time)
            print("server cost time: ", server_used_time)

        del trainX, trainY, Y
