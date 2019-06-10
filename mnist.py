import numpy as np
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


class LeNet5():
    def __init__(self, use_gpu = False):
        self.use_gpu = use_gpu
        self.model = ["None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None"]
    def load_weight(self, name, layer):
        c = np.load(name)
        layer.w, layer.b = c['w'], c['b']
        layer.w, layer.b = chainer.as_variable(layer.w), chainer.as_variable(layer.b)

    def save_weight(self, name, layer):
        print('process {} completed!'.format(name))
        np.savez(name, w = layer.w.array, b = layer.b.array)
        # pass

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
    def save_output(self, name, output):
        np.savez(name, x = output.array)
        pass

    def forward(self, out, process_layers):
        max_pool_layer = [2, 4]
        for i in process_layers:
            ts1 = time.time()
            out = self.model[i].forward(out)
            ts2 = time.time()
            print("layer {} forward time {}".format(i, (ts2-ts1)*1000.))
            name = "{}_output.npz".format(i)
            # name = 'init_wb/{}.npz'.format(i)
            self.save_output(name, out)
            # if i not in max_pool_layer:
            #     self.save_weight(name, self.model[i])
        return out
    def cal_gradients(self, d_out, process_layers, Y=None):
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
            print('layer {} cal gradients time {}'.format(i, (ts2-ts1)*1000))

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
        max_pool_layer = [2, 4]
        if layer_num not in max_pool_layer:
            model[layer_num].accumulate_params_grad(grads_w, grads_bias)
    def update_layers(self, process_layers, batch_size=128):
        max_pool_layer = [2, 4]
        for i in process_layers:
            if i not in max_pool_layer:
                ts1 = time.time()
                self.model[i].update_parameters(batch=batch_size)
                ts2 = time.time()
                print('layer {} update layer time {}'.format(i, (ts2-ts1)*1000))

if __name__ == '__main__':
    use_gpu = False
    lenet= LeNet5(use_gpu=use_gpu)
    process_layers = np.arange(1, 7+1)
    lenet.init_layers(process_layers)
    generations = 10
    batch_size = 512
    for i in range(generations):

        trainX, trainY = utils.get_batch_data(batch_size)
        Y = trainY.astype(np.int32)
        trainX = chainer.as_variable(trainX)
        Y = chainer.as_variable(Y)
        # print(trainX.shape)
        # np.savez('input.npz', x = trainX.array, y = Y.array)

        ts1 = time.time()
        temp_out = lenet.forward(trainX, process_layers)

        loss = F.softmax_cross_entropy(temp_out, Y)
        accuracy = F.accuracy(temp_out, Y)
        print('epoch #{} loss: {}'.format(i, loss))
        print('epoch #{} accuracy: {}'.format(i, accuracy))


        lenet.cal_gradients(temp_out, process_layers, Y)
        lenet.update_layers(process_layers)
        del trainX, trainY, Y
        ts2 = time.time()
        print('one opoch cost time: ', (ts2 - ts1) * 1000.)

