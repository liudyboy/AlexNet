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
import cifar_utils as utils
import gc
import time
import sys
import args
import _thread
import connect
from model import AlexNet


def process_gradients_exchange(alexnet, layer_num):
    global edge_address
    destination = edge_address
    max_pool_layer = [2, 4]
    if layer_num not in max_pool_layer:
        grads_w, grads_bias = alexnet.get_params_grads(layer_num)
        grads_w, grads_bias = get_one_layer_gradients(destination, grads_w, grads_bias, layer_num)
        alexnet.add_params_grads(layer_num, grads_w, grads_bias)
    return

def init_signal():
    global finish_flag
    finish_flag = 0                      # if fnish_flag equal 2 means one epoch training end
    return

def send_raw_data(threadName, threadID, destination, X, Y):
    global finish_flag
    ts1 = time.time()
    print(threadName, " start time: ", ts1)
    connect.conn_send_raw_data(destination, X.array, Y.array)
    print(threadName, " end")
    finish_flag += 1
    return

def send_output_data(destination, output, Y):
    ts1 = time.time()
    print('Send output to edge start time: ', ts1)
    reply = connect.conn_send_device_output_data(destination, output.array, Y.array)
    print('receive output gradients')
    return chainer.as_variable(reply)

def get_one_layer_gradients(destination, grads_w, grads_bias, layer_num):
    grads_w = chainer.as_variable(grads_w)
    grads_bias = chainer.as_variable(grads_bias)
    print("get layer {} gradients".format(layer_num))
    recv_grads_w, recv_grads_bias = connect.conn_get_gradients(destination, grads_w.array, grads_bias.array, layer_num, 'device')
    return chainer.as_variable(recv_grads_w), chainer.as_variable(recv_grads_bias)

# just in case to wait two new threads complete
def wait_threads_complete(cloud):
    global finish_flag
    target = 1
    if cloud != 0:
        target += 1
    print('wait threads complete')
    while finish_flag is not target:
        pass
    print('one epoch training completed')


edge_address = "192.168.1.77:50055"
cloud_address = "192.168.1.70:50055"

if __name__ == "__main__":
    my_args = args.init_args()
    device_run_layers, cloud_run_layers = my_args.M1, my_args.M2
    alexnet =  AlexNet()
    alexnet.init_layers(np.arange(1, device_run_layers+1))
    edge_batch, cloud_batch = 128, 0
    generations = 10
    total_batch_size = 128
    for i in range(generations):
        init_signal()
        trainX, trainY = utils.get_batch_data(total_batch_size)
        trainX = chainer.as_variable(trainX)
        trainY = chainer.as_variable(trainY.astype(np.int32))

        ts1 = time.time()

        #spilt raw data for edge , cloud
        batch_size = trainX.shape[0]
        cloud_X = trainX[batch_size-cloud_batch:]
        cloud_Y = trainY[batch_size-cloud_batch:]
        trainX = trainX[:batch_size-cloud_batch]
        trainY = trainY[:batch_size-cloud_batch]
        batch_size = trainX.shape[0]
        edge_X = trainX[batch_size-edge_batch:]
        edge_Y = trainY[batch_size-edge_batch:]
        trainX = trainX[:batch_size-edge_batch]
        trainY = trainY[:batch_size-edge_batch]

        try:
            if cloud_run_layers != 0:
                _thread.start_new_thread(send_raw_data, ("send cloud thread", 0, cloud_address, cloud_X, cloud_Y))
            _thread.start_new_thread(send_raw_data, ("send edge thread", 1, edge_address, edge_X, edge_Y))
        except:
            print('send raw thread error')
        if device_run_layers != 0:
            ts1 = time.time()
            process_layers = np.arange(1, device_run_layers+1)
            output = alexnet.forward(trainX, process_layers)

            tts2 = time.time()
            print("start send device output data time: ", tts2)
            output_reply = send_output_data(edge_address, output, trainY)

            tts3 = time.time()
            alexnet.cal_gradients(output_reply, process_layers, trainY)

            tts4 = time.time()
            print('cal gradients cost time: ', (tts4 - tts3) * 1000.)

            for j in process_layers:
                process_gradients_exchange(alexnet, j)
                alexnet.update_one_layer_parameters(j, total_batch_size)


        wait_threads_complete(cloud_run_layers)

        ts2 = time.time()

        one_epoch_time = (ts2 - ts1) * 1000.
        print('one epoch training time: ', one_epoch_time)


