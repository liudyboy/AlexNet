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
import connect
import _thread


def init_layers(process_layers):
    global conv1, conv2, conv3, conv4, conv5, fc6, fc7, fc8, max_pool1, max_pool2, max_pool5
    conv_stride = [4, 4]
    paralle = False
    conv_stride = [4, 4]
    if 1 in process_layers:
        self.conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride, paralle=paralle)
        c = np.load("init_wb/conv1.npz")
        self.conv1.w, self.conv1.b = c['w'], c['b']
        self.conv1.w, self.conv1.b = chainer.as_variable(self.conv1.w), chainer.as_variable(self.conv1.b)
    if 2 in process_layers:
        self.max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
    if 3 in process_layers:
        self.conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1], paralle=paralle)
        c = np.load("init_wb/conv2.npz")
        self.conv2.w, self.conv2.b = c['w'], c['b']
        self.conv2.w, self.conv2.b = chainer.as_variable(self.conv2.w), chainer.as_variable(self.conv2.b)
    if 4 in process_layers:
        self.max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
    if 5 in process_layers:
        self.conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1], paralle=paralle)
        c = np.load("init_wb/conv3.npz")
        self.conv3.w, self.conv3.b = c['w'], c['b']
        self.conv3.w, self.conv3.b = chainer.as_variable(self.conv3.w), chainer.as_variable(self.conv3.b)
    if 6 in process_layers:
        self.conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1], paralle=paralle)
        c = np.load("init_wb/conv4.npz")
        self.conv4.w, self.conv4.b = c['w'], c['b']
        self.conv4.w, self.conv4.b = chainer.as_variable(self.conv4.w), chainer.as_variable(self.conv4.b)
    if 7 in process_layers:
        self.conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1], paralle=paralle)
        c = np.load("init_wb/conv5.npz")
        self.conv5.w, self.conv5.b = c['w'], c['b']
        self.conv5.w, self.conv5.b = chainer.as_variable(self.conv5.w), chainer.as_variable(self.conv5.b)
    if 8 in process_layers:
        self.max_pool5 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
    if 9 in process_layers:
        self.fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6', paralle=paralle)
        c = np.load("init_wb/fc6.npz")
        self.fc6.w, self.fc6.b = c['w'], c['b']
        self.fc6.w, self.fc6.b = chainer.as_variable(self.fc6.w), chainer.as_variable(self.fc6.b)
    if 10 in process_layers:
        self.fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7', paralle=paralle)
        c = np.load("init_wb/fc7.npz")
        self.fc7.w, self.fc7.b = c['w'], c['b']
        self.fc7.w, self.fc7.b = chainer.as_variable(self.fc7.w), chainer.as_variable(self.fc7.b)
    if 11 in process_layers:
        self.fc8 = layers.dense(200, activation='relu', name='fc8', paralle=paralle)
        c = np.load("init_wb/fc8.npz")
        self.fc8.w, self.fc8.b = c['w'], c['b']
        self.fc8.w, self.fc8.b = chainer.as_variable(self.fc8.w), chainer.as_variable(self.fc8.b)


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

def cal_gradients(d_out, Y, process_layers):
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



edge_address = "192.168.1.77:50051"
cloud_address = ""


def send_raw_data(threadName, threadID, destination, X, Y):
    global finish_sinal
    print(threadName, " start")
    connect.conn_send_input(destination, X, Y)
    print(threadName, " end")
    finish = True

def send_output_data(threadName, threadID, destination, X, Y):
    global recv_output_reply_flag, output_reply
    print(threadName, ' start')
    output_reply = connect.conn_device_send_output(destination, X, Y)
    output_reply = chainer.as_variable(output_reply)
    recv_output_reply_flag = True
    print(threadName, ' end')
def wait_reply_output():
    global recv_output_reply_flag
    while recv_output_reply_flag is False:
        pass

def wait_epoch_end():
    global finish_sinal
    while finish_sinal is False:
        pass
if __name__ == "__main__":
    # initial layers run in device
    device_run_layers, cloud_run_layers = args.args_prase()
    init_layers(np.arange(device_run_layers+1))

    edge_batch, cloud_batch = 50, 50
    logging.basicConfig()
    ts = time.time()
    start_time = time.ctime(ts)
    print("start time:", start_time)
    generations = 100
    batch_size = 128
    for i in range(generations):
        global recv_output_reply_flag = False
        global output_reply, finish_sinal = False

        trainX, trainY = utils.get_batch_data(batch_size)
        trainY = trainY.astype(np.int32)

        #spilt raw data for edge , cloud
        cloud_X = trainX[batch_size-cloud_batch:]
        cloud_Y = trainY[batch_size-cloud_batch:]
        trainX = trainX[:batch_size-cloud_batch]
        trainY = trainY[:batch_size=cloud_batch]

        batch_size = trainX.shape[0]
        edge_X = trainX[batch_size-edge_batch:]
        edge_Y = trainY[batch_size-edge_batch:]
        trainX = trainX[:batch_size-edge_batch]
        trainY = trainY[:batch_size-edge_batch]

        try:
            _thread.start_new_thread(send_raw_data, ("send cloud thread", 0, cloud_address, cloud_X, cloud_Y))
            _thread.start_new_thread(send_raw_data, ("send edge thread", 1, edge_address, edge_X, edge_Y))
        except:
            print('thread error')

        ts1 = time.time()
        trainX = chainer.as_variable(trainX)
        Y = trainY.astype(np.int32)
        Y = chainer.as_variable(Y)
        process_layers = np.arange(device_run_layers+1)
        if device_run_layers > 0:
            output = Forward(trainX, process_layers)
            try:
                _thread.start_new_thread(send_output_data, ('send output thread', 2, edge_address, output, Y))
            except:
                print('send output thread error')

            wait_reply_output()
            process_layers = np.arange(device_run_layers+1)
            cal_gradients(output_reply, Y, process_layers)
        wait_epoch_end()
        del trainX, trainY, Y
