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
import mnist_utils as utils
import gc
import time
import sys
import args
import _thread
import connect

class LeNet5():
    def __init__(self, use_gpu = False):
        self.batch_size = 512
        self.use_gpu = use_gpu
        self.model = ["None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None"]
    def load_weight(self, name, layer):
        c = np.load(name)
        layer.w, layer.b = c['w'], c['b']
        layer.w, layer.b = chainer.as_variable(layer.w), chainer.as_variable(layer.b)

    def init_layers(self, process_layers):
        if 1 in process_layers:
            self.conv1 = layers.conv2d(filters=6, kernel=[5, 5], padding='SAME', name='conv1', activation='tanh', use_gpu=self.use_gpu)
            self.load_weight("init_wb/1.npz", self.conv1)
            self.model[1] = self.conv1
        if 2 in process_layers:
            self.max_pool1 = layers.max_pool2d(ksize=[2, 2], stride=[2, 2], name='max_pool1', use_gpu=self.use_gpu)
            self.model[2] = self.max_pool1
        if 3 in process_layers:
            self.conv2 = layers.conv2d(filters=16, kernel=[5, 5], padding='SAME', name='conv3', activation='tanh', use_gpu=self.use_gpu)
            self.load_weight("init_wb/3.npz", self.conv2)
            self.model[3] = self.conv2
        if 4 in process_layers:
            self.max_pool2 = layers.max_pool2d(ksize=[2, 2], stride=[2, 2], name='max_pool1', use_gpu=self.use_gpu)
            self.model[4] = self.max_pool2

        if 5 in process_layers:
            self.conv3 = layers.conv2d(filters=120, kernel=[5, 5], padding='SAME', name='conv3', activation='tanh', use_gpu=self.use_gpu)
            self.load_weight("init_wb/5.npz", self.conv3)
            self.model[5] = self.conv3

        if 6 in process_layers:
            self.fc1 = layers.dense(out_size=84, activation='tanh', name='fc1', use_gpu=self.use_gpu)
            self.load_weight("init_wb/6.npz", self.fc1)
            self.model[6] = self.fc1
        if 7 in process_layers:
            self.fc2 = layers.dense(out_size=10, activation='softmax', name='fc2', use_gpu=self.use_gpu)
            self.load_weight("init_wb/7.npz", self.fc2)
            self.model[7] = self.fc2
    def forward(self, out, process_layers):
        if self.use_gpu:
            out.to_gpu(0)
        for i in process_layers:
            ts1 = time.time()
            out = self.model[i].forward(out)
            ts2 = time.time()
            print('layer {} forward time: {}'.format(i, (ts2-ts1)*1000))
        return out
    def cal_gradients(self, d_out, process_layers, Y=None):
        if self.use_gpu:
            d_out.to_gpu(0)
        process_layers = np.flip(process_layers, axis=0)
        for i in process_layers:
            ts1 = time.time()
            if i == 7:
                loss = F.softmax_cross_entropy(d_out, Y)
                accuracy = F.accuracy(d_out, Y)
                print('loss: {}'.format(loss))
                print('accuracy: {}'.format(accuracy))
                d_out = chainer.grad([loss], [d_out])
                if isinstance(d_out, (list)):
                    d_out = d_out[0]
            d_out = self.model[i].backward(d_out)
            ts2 = time.time()
            print('layer {} cal graident time: {}'.format(i, (ts2 - ts1) *1000))
        return d_out
    def update_one_layer_parameters(self, layer_num, batch_size):
        max_pool_layer = [2, 4]
        if layer_num not in max_pool_layer:
            self.model[layer_num].update_parameters(batch=batch_size)

    def get_params_grads(self, layer_num):
        max_pool_layer = [2, 4]
        if layer_num not in max_pool_layer:
            grads_w, grads_bias = self.model[layer_num].get_params_grad()
            return grads_w, grads_bias
    def add_params_grads(self, layer_num, grads_w, grads_bias):
        if self.use_gpu:
            grads_w.to_gpu(0)
            grads_bias.to_gpu(0)
        max_pool_layer = [2, 4]
        if layer_num not in max_pool_layer:
            self.model[layer_num].accumulate_params_grad(grads_w, grads_bias)
    def update_layers_parameters(self, process_layers, batch_size=512):
        max_pool_layer = [2, 4]
        for i in process_layers:
            if i not in max_pool_layer:
                self.model[i].update_parameters(batch=batch_size)


class AlexNet():
    def __init__(self, use_gpu = False):
        self.use_gpu = use_gpu
        self.batch_size = 128
        self.model = ["None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None"]

    def load_weight(self, name, layer):
        c = np.load(name)
        layer.w, layer.b = c['w'], c['b']
        layer.w, layer.b = chainer.as_variable(layer.w), chainer.as_variable(layer.b)

    def init_layers(self, process_layers):
        if 1 in process_layers:
            self.conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=[4, 4], use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv1.npz", self.conv1)
            self.model[1] = self.conv1
        if 2 in process_layers:
            self.max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2], use_gpu=self.use_gpu)
            self.model[2] = self.max_pool1
        if 3 in process_layers:
            self.conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1], use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv2.npz", self.conv2)
            self.model[3] = self.conv2
        if 4 in process_layers:
            self.max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2], use_gpu=self.use_gpu)
            self.model[4] = self.max_pool2
        if 5 in process_layers:
            self.conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1], use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv3.npz", self.conv3)
            self.model[5] = self.conv3
        if 6 in process_layers:
            self.conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1], use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv4.npz", self.conv4)
            self.model[6] = self.conv4
        if 7 in process_layers:
            self.conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1], use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv5.npz", self.conv5)
            self.model[7] = self.conv5
        if 8 in process_layers:
            self.max_pool5 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2], use_gpu=self.use_gpu)
            self.model[8] = self.max_pool5
        if 9 in process_layers:
            self.fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6', use_gpu=self.use_gpu)
            self.load_weight("init_wb/fc6.npz", self.fc6)
            self.model[9] = self.fc6
        if 10 in process_layers:
            self.fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7', use_gpu=self.use_gpu)
            self.load_weight("init_wb/fc7.npz", self.fc7)
            self.model[10] = self.fc7
        if 11 in process_layers:
            self.fc8 = layers.dense(200, activation='relu', name='fc8', use_gpu=self.use_gpu)
            self.load_weight("init_wb/fc8.npz", self.fc8)
            self.model[11] = self.fc8

    def forward(self, out, process_layers):
        if self.use_gpu:
            out.to_gpu(0)
        for i in process_layers:
            ts1 = time.time()
            out = self.model[i].forward(out)
            ts2 = time.time()
            print('layer {} forward time: {}'.format(i, (ts2-ts1)*1000))
        return out
    def cal_gradients(self, d_out, process_layers, Y=None):
        if self.use_gpu:
            d_out.to_gpu(0)
        process_layers = np.flip(process_layers, axis=0)
        for i in process_layers:
            ts1 = time.time()
            if i == 11:
                loss = F.softmax_cross_entropy(d_out, Y)
                accuracy = F.accuracy(d_out, Y)
                print('loss: {}'.format(loss))
                print('accuracy: {}'.format(accuracy))
                d_out = chainer.grad([loss], [d_out])
                if isinstance(d_out, (list)):
                    d_out = d_out[0]
            d_out = self.model[i].backward(d_out)
            ts2 = time.time()
            print('layer {} cal graident time: {}'.format(i, (ts2 - ts1) *1000))
        return d_out
    def update_one_layer_parameters(self, layer_num, batch_size=128):
        max_pool_layer = [2, 4, 8]
        if layer_num not in max_pool_layer:
            self.model[layer_num].update_parameters(batch=batch_size)

    def get_params_grads(self, layer_num):
        max_pool_layer = [2, 4, 8]
        if layer_num not in max_pool_layer:
            grads_w, grads_bias = self.model[layer_num].get_params_grad()
            return grads_w, grads_bias
    def add_params_grads(self, layer_num, grads_w, grads_bias):
        if self.use_gpu:
            grads_w.to_gpu(0)
            grads_bias.to_gpu(0)
        max_pool_layer = [2, 4, 8]
        if layer_num not in max_pool_layer:
            self.model[layer_num].accumulate_params_grad(grads_w, grads_bias)
    def update_layers_parameters(self, process_layers, batch_size=128):
        max_pool_layer = [2, 4, 8]
        for i in process_layers:
            if i not in max_pool_layer:
                self.model[i].update_parameters(batch=batch_size)
class VGG16():
    def __init__(self, use_gpu = False):
        self.batch_size = 256
        self.target_size = 200
        self.generations = 100
        self.use_gpu = use_gpu

        self.conv3_64_features = 64
        self.conv3_64_kernel = [3, 3]

        self.conv3_128_features = 128
        self.conv3_128_kernel = [3, 3]

        self.conv3_256_features = 256
        self.conv3_256_kernel = [3, 3]

        self.conv3_512_features = 512
        self.conv3_512_kernel = [3, 3]

        self.max_pool_window = [2, 2]
        self.max_pool_stride = [2, 2]

        self.fc_4096 = 4096
        self.fc_1000 = 1000
        self.model = ["None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None"]
    def load_weight(self, name, layer):
        c = np.load(name)
        layer.w, layer.b = c['w'], c['b']
        layer.w, layer.b = chainer.as_variable(layer.w), chainer.as_variable(layer.b)

    def init_layers(self, process_layers):
        if 1 in process_layers:
            self.conv1 = layers.conv2d(filters=self.conv3_64_features, kernel=self.conv3_64_kernel, padding='SAME', name='conv1', activation='relu', use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv1.npz", self.conv1)
            self.model[1] = self.conv1
        if 2 in process_layers:
            self.conv2 = layers.conv2d(filters=self.conv3_64_features, kernel=self.conv3_64_kernel, padding='SAME', name='conv2', activation='relu', use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv2.npz", self.conv2)
            self.model[2] = self.conv2
        if 3 in process_layers:
            self.max_pool1 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool1', use_gpu=self.use_gpu)
            self.load_weight("init_wb/max_pool1.npz", self.max_pool1)
            self.model[3] = self.max_pool1

        if 4 in process_layers:
            self.conv3 = layers.conv2d(filters=self.conv3_128_features, kernel=self.conv3_128_kernel, padding='SAME', name='conv3', activation='relu', use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv3.npz", self.conv3)
            self.model[4] = self.conv3
        if 5 in process_layers:
            self.conv4 = layers.conv2d(filters=self.conv3_128_features, kernel=self.conv3_128_kernel, padding='SAME', name='conv4', activation='relu', use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv4.npz", self.conv4)
            self.model[5] = self.conv4
        if 6 in process_layers:
            self.max_pool2 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool2', use_gpu=self.use_gpu)
            self.load_weight("init_wb/max_pool2.npz", self.max_pool2)
            self.model[6] = self.max_pool2

        if 7 in process_layers:
            self.conv5 = layers.conv2d(filters=self.conv3_256_features, kernel=self.conv3_256_kernel, padding='SAME', name='conv5', activation='relu', use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv5.npz", self.conv5)
            self.model[7] = self.conv5

        if 8 in process_layers:
            self.conv6 = layers.conv2d(filters=self.conv3_256_features, kernel=self.conv3_256_kernel, padding='SAME', name='conv6', activation='relu', use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv6.npz", self.conv6)
            self.model[8] = self.conv6
        if 9 in process_layers:
            self.conv7 = layers.conv2d(filters=self.conv3_256_features, kernel=self.conv3_256_kernel, padding='SAME', name='conv7', activation='relu', use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv7.npz", self.conv7)
            self.model[9] = self.conv7
        if 10 in process_layers:
            self.max_pool3 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool3', use_gpu=self.use_gpu)
            self.load_weight("init_wb/max_pool3.npz", self.max_pool3)
            self.model[10] = self.max_pool3

        if 11 in process_layers:
            self.conv8 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv8', activation='relu', use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv8.npz", self.conv8)
            self.model[11] = self.conv8
        if 12 in process_layers:
            self.conv9 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv9', activation='relu', use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv9.npz", self.conv9)
            self.model[12] = self.conv9
        if 13 in process_layers:
            self.conv10 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv10', activation='relu', use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv10.npz", self.conv10)
            self.model[13] = self.conv10
        if 14 in process_layers:
            self.max_pool4 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool4', use_gpu=self.use_gpu)
            self.load_weight("init_wb/max_pool4.npz", self.max_pool4)
            self.model[14] = self.max_pool4

        if 15 in process_layers:
            self.conv11 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv11', activation='relu', use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv11.npz", self.conv11)
            self.model[15] = self.conv11
        if 16 in process_layers:
            self.conv12 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv12', activation='relu', use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv12.npz", self.conv12)
            self.model[16] = self.conv12
        if 17 in process_layers:
            self.conv13 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv13', activation='relu', use_gpu=self.use_gpu)
            self.load_weight("init_wb/conv13.npz", self.conv13)
            self.model[17] = self.conv13
        if 18 in process_layers:
            self.max_pool5 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool5', use_gpu=self.use_gpu)
            self.load_weight("init_wb/max_pool5.npz", self.max_pool5)
            self.model[18] = self.max_pool5
        if 19 in process_layers:
            self.fc1 = layers.dense(out_size=self.fc_4096, activation='relu', name='fc1', use_gpu=self.use_gpu)
            self.load_weight("init_wb/fc1.npz", self.fc1)
            self.model[19] = self.fc1
        if 20 in process_layers:
            self.fc2 = layers.dense(out_size=self.fc_4096, activation='relu', name='fc2', use_gpu=self.use_gpu)
            self.load_weight("init_wb/fc2.npz", self.fc2)
            self.model[20] = self.fc2
        if 21 in process_layers:
            self.fc3 = layers.dense(out_size=self.target_size, activation='relu', name='fc3', use_gpu=self.use_gpu)
            self.load_weight("init_wb/fc3.npz", self.fc3)
            self.model[21] = self.fc3

    def forward(self, out, process_layers):
        for i in process_layers:
            out = self.model[i].forward(out)
        return out
    def cal_gradients(self, d_out, process_layers, Y=None):
        process_layers = np.flip(process_layers, axis=0)
        for i in process_layers:
            if i == 21:
                loss = F.softmax_cross_entropy(d_out, Y)
                accuracy = F.accuracy(d_out, Y)
                print('loss: {}'.format(loss))
                print('accuracy: {}'.format(accuracy))
                d_out = chainer.grad([loss], [d_out])
                if isinstance(d_out, (list)):
                    d_out = d_out[0]
            d_out = self.model[i].backward(d_out)
        return d_out
    def update_one_layer_parameters(self, layer_num, batch_size):
        max_pool_layer = [3, 6, 10, 14, 18]
        if layer_num not in max_pool_layer:
            self.model[layer_num].update_parameters(batch=batch_size)

    def get_params_grads(self, layer_num):
        max_pool_layer = [3, 6, 10, 14, 18]
        if layer_num not in max_pool_layer:
            grads_w, grads_bias = self.model[layer_num].get_params_grad()
            return grads_w, grads_bias
    def add_params_grads(self, ler_num, grads_w, grads_bias):
        max_pool_layer = [3, 6, 10, 14, 18]
        if layer_num not in max_pool_layer:
            self.model[layer_num].accumulate_params_grad(grads_w, grads_bias)
