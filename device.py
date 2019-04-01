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
import args



def Forward(out, process_layers):
    if 1 in process_layers:
        out = conv1.forward(out)
    if 2 in process_layers:
        out = max_pool1.forward(out)
    if 3 in process_layers:
        out = conv2.forward(out)
    if 4 in process_layers:
        out = max_pool2.forward(out)
    if 5 in process_layers:
        out = conv3.forward(out)
    if 6 in process_layers:
        out = conv4.forward(out)
    if 7 in process_layers:
        out = conv5.forward(out)
    if 8 in process_layers:
        out = max_pool5.forward(out)
    if 9 in process_layers:
        out = fc6.forward(out)
    if 10 in process_layers:
        out = fc7.forward(out)
    if 11 in process_layers:
        out = fc8.forward(out)
    return out

def Backward(d_out, Y, process_layers):

    if 11 in process_layers:
        loss = F.softmax_cross_entropy(d_out, Y)
        accuracy = F.accuracy(d_out, Y)
        print('loss: {}'.format(loss))
        print('accuracy: {}'.format(accuracy))

        d_out = chainer.grad([loss], [d_out])
        if isinstance(d_out, (list)):
            d_out = d_out[0]
        d_out = fc8.backward(d_out)
    if 10 in process_layers:
        d_out = fc7.backward(d_out)
    if 9 in process_layers:
        d_out = fc6.backward(d_out)
    if 8 in process_layers:
        d_out = max_pool5.backward(d_out)
    if 7 in process_layers:
        d_out = conv5.backward(d_out)
    if 6 in process_layers:
        d_out = conv4.backward(d_out)
    if 5 in process_layers:
        d_out = conv3.backward(d_out)
    if 4 in process_layers:
        d_out = max_pool2.backward(d_out)
    if 3 in process_layers:
        d_out = conv2.backward(d_out)
    if 2 in process_layers:
        d_out = max_pool1.backward(d_out)
    if 1 in process_layers:
        d_out = conv1.backward(d_out)

    del d_out



def run(sendArray, Y):
    with grpc.insecure_channel("192.168.1.77:50051", options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)]) as channel:
        stub = communication_pb2_grpc.CommStub(channel)
        sendArray = pickle.dumps(sendArray)
        Y = pickle.dumps(Y)
        recv_array = stub.Forwarding(communication_pb2.ArrayRecv(array=sendArray, Y=Y))
        recv_array = pickle.loads(recv_array.array)
    return recv_array

def init_layers(process_layers):
    global conv1, conv2, conv3, conv4, conv5, fc6, fc7, fc8, max_pool1, max_pool2, max_pool5
    conv_stride = [4, 4]

    if 1 in process_layers:
        conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride)
    if 2 in process_layers:
        max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
    if 3 in process_layers:
        conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1])
    if 4 in process_layers:
        max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
    if 5 in process_layers:
        conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1])
    if 6 in process_layers:
        conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1])
    if 7 in process_layers:
        conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1])
    if 8 in process_layers:
        max_pool5 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
    if 9 in process_layers:
        fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6')
    if 10 in process_layers:
        fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7')
    if 11 in process_layers:
        fc8 = layers.dense(1000, activation='relu', name='fc8')




if __name__ == "__main__":

    process_layers = args.args_prase()

    init_layers(process_layers)

    logging.basicConfig()
    ts = time.time()
    start_time = time.ctime(ts)
    print("start time:", start_time)
    generations = 100
    batch_size = 64


    for i in range(generations):
        trainX, trainY = utils.get_batch_data(batch_size)

        ts1 = time.time()
        trainX = chainer.as_variable(trainX)
        Y = trainY.astype(np.int32)
        Y = chainer.as_variable(Y)

        if 0 in process_layers:
            output = trainX
        else:
            output = Forward(trainX, process_layers)

        ts3 = time.time()
        output = run(output, Y)

        ts4 = time.time()
        Backward(output, Y, process_layers)

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
            print("device computing time: ", client_used_time)
            print("others cost time: ", server_used_time)

        del trainX, trainY, Y
