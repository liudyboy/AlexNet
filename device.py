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

def init_layers(process_layers):
    global conv1, conv2, conv3, conv4, conv5, fc6, fc7, fc8, max_pool1, max_pool2, max_pool5
    conv_stride = [4, 4]
    conv_stride = [4, 4]
    if 1 in process_layers:
        conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride )
        c = np.load("init_wb/conv1.npz")
        conv1.w, conv1.b = c['w'], c['b']
        conv1.w, conv1.b = chainer.as_variable(conv1.w), chainer.as_variable(conv1.b)
    if 2 in process_layers:
        max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
    if 3 in process_layers:
        conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1] )
        c = np.load("init_wb/conv2.npz")
        conv2.w, conv2.b = c['w'], c['b']
        conv2.w, conv2.b = chainer.as_variable(conv2.w), chainer.as_variable(conv2.b)
    if 4 in process_layers:
        max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
    if 5 in process_layers:
        conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1] )
        c = np.load("init_wb/conv3.npz")
        conv3.w, conv3.b = c['w'], c['b']
        conv3.w, conv3.b = chainer.as_variable(conv3.w), chainer.as_variable(conv3.b)
    if 6 in process_layers:
        conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1] )
        c = np.load("init_wb/conv4.npz")
        conv4.w, conv4.b = c['w'], c['b']
        conv4.w, conv4.b = chainer.as_variable(conv4.w), chainer.as_variable(conv4.b)
    if 7 in process_layers:
        conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1] )
        c = np.load("init_wb/conv5.npz")
        conv5.w, conv5.b = c['w'], c['b']
        conv5.w, conv5.b = chainer.as_variable(conv5.w), chainer.as_variable(conv5.b)
    if 8 in process_layers:
        max_pool5 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
    if 9 in process_layers:
        fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6' )
        c = np.load("init_wb/fc6.npz")
        fc6.w, fc6.b = c['w'], c['b']
        fc6.w, fc6.b = chainer.as_variable(fc6.w), chainer.as_variable(fc6.b)
    if 10 in process_layers:
        fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7' )
        c = np.load("init_wb/fc7.npz")
        fc7.w, fc7.b = c['w'], c['b']
        fc7.w, fc7.b = chainer.as_variable(fc7.w), chainer.as_variable(fc7.b)
    if 11 in process_layers:
        fc8 = layers.dense(200, activation='relu', name='fc8' )
        c = np.load("init_wb/fc8.npz")
        fc8.w, fc8.b = c['w'], c['b']
        fc8.w, fc8.b = chainer.as_variable(fc8.w), chainer.as_variable(fc8.b)


def forward(out, process_layers):
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

def cal_gradients(d_out, process_layers, Y=None):
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

def update_one_layer_parameters(layer_num, batch_size):
    if 1 == layer_num:
        conv1.update_parameters(batch=batch_size)
    if 2 == layer_num:
        pass          # it is maxpool layer needn't exchange gradients
    if 3 == layer_num:
        conv2.update_parameters(batch=batch_size)
    if 4 == layer_num:
        pass          # it is maxpool layer needn't exchange gradients
    if 5 == layer_num:
        conv3.update_parameters(batch=batch_size)
    if 6 == layer_num:
        conv4.update_parameters(batch=batch_size)
    if 7 == layer_num:
        conv5.update_parameters(batch=batch_size)
    if 8 == layer_num:
        pass          # it is maxpool layer needn't exchange gradients
    if 9 == layer_num:
        fc6.update_parameters(batch=batch_size)
    if 10 == layer_num:
        fc7.update_parameters(batch=batch_size)
    if 11 == layer_num:
        fc8.update_parameters(batch=batch_size)
    return

def process_gradients_exchange(layer_num):
    global edge_address
    destination = edge_address
    if 1 == layer_num:
        grads_w, grads_bias = conv1.get_params_grad()
        grads_w, grads_bias = get_one_layer_gradients(destination, grads_w, grads_bias, layer_num)
        conv1.accumulate_params_grad(grads_w, grads_bias)
    if 2 == layer_num:
        pass          # it is maxpool layer needn't exchange gradients
    if 3 == layer_num:
        grads_w, grads_bias = conv2.get_params_grad()
        grads_w, grads_bias = get_one_layer_gradients(destination, grads_w, grads_bias, layer_num)
        conv2.accumulate_params_grad(grads_w, grads_bias)
    if 4 == layer_num:
        pass          # it is maxpool layer needn't exchange gradients
    if 5 == layer_num:
        grads_w, grads_bias = conv3.get_params_grad()
        grads_w, grads_bias = get_one_layer_gradients(destination, grads_w, grads_bias, layer_num)
        conv3.accumulate_params_grad(grads_w, grads_bias)
    if 6 == layer_num:
        grads_w, grads_bias = conv4.get_params_grad()
        grads_w, grads_bias = get_one_layer_gradients(destination, grads_w, grads_bias, layer_num)
        conv4.accumulate_params_grad(grads_w, grads_bias)
    if 7 == layer_num:
        grads_w, grads_bias = conv5.get_params_grad()
        grads_w, grads_bias = get_one_layer_gradients(destination, grads_w, grads_bias, layer_num)
        conv5.accumulate_params_grad(grads_w, grads_bias)
    if 8 == layer_num:
        pass          # it is maxpool layer needn't exchange gradients
    if 9 == layer_num:
        grads_w, grads_bias = fc6.get_params_grad()
        grads_w, grads_bias = get_one_layer_gradients(destination, grads_w, grads_bias, layer_num)
        fc6.accumulate_params_grad(grads_w, grads_bias)
    if 10 == layer_num:
        grads_w, grads_bias = fc7.get_params_grad()
        grads_w, grads_bias = get_one_layer_gradients(destination, grads_w, grads_bias, layer_num)
        fc7.accumulate_params_grad(grads_w, grads_bias)
    if 11 == layer_num:
        grads_w, grads_bias = fc8.get_params_grad()
        grads_w, grads_bias = get_one_layer_gradients(destination, grads_w, grads_bias, layer_num)
        fc8.accumulate_params_grad(grads_w, grads_bias)

def init_signal():
    global finish_flag
    finish_flag = 0                      # if fnish_flag equal 2 means one epoch training end

def send_raw_data(threadName, threadID, destination, X, Y):
    global finish_flag
    print(threadName, " start")
    connect.conn_send_raw_data(destination, X.array, Y.array)
    print(threadName, " end")
    finish_flag += 1

def send_output_data(destination, output, Y):
    print('Send output to edge')
    reply = connect.conn_send_device_output_data(destination, output.array, Y.array)
    print('receive output gradients')
    return chainer.as_variable(reply)

def get_one_layer_gradients(destination, grads_w, grads_bias, layer_num):
    # print('get layer ', layer_num, ' gradients')
    while True:
        recv_grads_w, recv_grads_bias = connect.conn_get_gradients(destination, grads_w, grads_bias, layer_num)
        if recv_grads_w.shape[0] != 1:
            break;
    # print('complete get layer', layer_num, ' gradients')
    # print('layer gradients: ', recv_grads_w.shape)
    return chainer.as_variable(recv_grads_w), chainer.as_variable(recv_grads_bias)

# just in case to wait two new threads complete
def wait_threads_complete():
    print('wait threads complete')
    while finish_flag is not 2:
        pass
    print('one epoch training completed')


edge_address = "192.168.1.77:50055"
cloud_address = "192.168.1.153:50052"

if __name__ == "__main__":

    device_run_layers, cloud_run_layers = args.args_prase()
    init_layers(np.arange(device_run_layers+1))
    edge_batch, cloud_batch = 75, 20
    logging.basicConfig()
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
            _thread.start_new_thread(send_raw_data, ("send cloud thread", 0, cloud_address, cloud_X, cloud_Y))
            _thread.start_new_thread(send_raw_data, ("send edge thread", 1, edge_address, edge_X, edge_Y))
        except:
            print('send raw thread error')


        process_layers = np.arange(1, device_run_layers+1)
        output = forward(trainX, process_layers)

        output_reply = send_output_data(edge_address, output, trainY)

        cal_gradients(output_reply, process_layers, trainY)

        for j in process_layers:
            process_gradients_exchange(j)
            update_one_layer_parameters(j, total_batch_size)

        wait_threads_complete()

        ts2 = time.time()

        one_epoch_time = (ts2 - ts1) * 1000.
        print('one epoch training time: ', one_epoch_time)


