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
import gc
import time
import sys
import args



def init_layers(process_layers):
    global conv1, conv2, conv3, conv4, conv5, fc6, fc7, fc8, max_pool1, max_pool2, max_pool5
    conv_stride = [4, 4]

    if 1 in process_layers:
        conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=1)
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
    inputx = np.arange(4*3*4*4).reshape((4, 3, 4, 4))
    inputx = chainer.as_variable(np.asarray(inputx, dtype=np.float32))

    input1 = inputx[2:]
    print(input1.array)
    print("-------------")
    inputx = inputx[0:2]
    print(inputx.array)
    # conv1 = layers.conv2d(filters=4, kernel=[2, 2], padding='SAME', name='conv1', activation='relu', stride=[1, 1])
    # conv1.w = np.ones(shape=(4, 3, 2, 2), dtype=np.float32)
    # conv1.b = np.zeros(shape=(4), dtype=np.float32)

    # output = conv1.forward(inputx)
    # print("------------------------")
    # print(output[0])
    # print('------------------')
    # print(output[1])

