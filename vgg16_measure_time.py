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
import tiny_utils as utils
import gc
import time
class VGG16():
    batch_size = 128
    target_size = 200
    generations = 100


    conv3_64_features = 64
    conv3_64_kernel = [3, 3]

    conv3_128_features = 128
    conv3_128_kernel = [3, 3]

    conv3_256_features = 256
    conv3_256_kernel = [3, 3]

    conv3_512_features = 512
    conv3_512_kernel = [3, 3]

    max_pool_window = [2, 2]
    max_pool_stride = [2, 2]

    fc_4096 = 4096
    fc_1000 = 1000

    generations = 0

    def __init__(self, use_gpu=False):
        self.conv1 = layers.conv2d(filters=self.conv3_64_features, kernel=self.conv3_64_kernel, padding='SAME', name='conv1', activation='relu', use_gpu=use_gpu)
        self.conv2 = layers.conv2d(filters=self.conv3_64_features, kernel=self.conv3_64_kernel, padding='SAME', name='conv2', activation='relu', use_gpu=use_gpu)
        self.max_pool1 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool1')

        self.conv3 = layers.conv2d(filters=self.conv3_128_features, kernel=self.conv3_128_kernel, padding='SAME', name='conv3', activation='relu', use_gpu=use_gpu)
        self.conv4 = layers.conv2d(filters=self.conv3_128_features, kernel=self.conv3_128_kernel, padding='SAME', name='conv4', activation='relu', use_gpu=use_gpu)
        self.max_pool2 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool2')

        self.conv5 = layers.conv2d(filters=self.conv3_256_features, kernel=self.conv3_256_kernel, padding='SAME', name='conv5', activation='relu', use_gpu=use_gpu)
        self.conv6 = layers.conv2d(filters=self.conv3_256_features, kernel=self.conv3_256_kernel, padding='SAME', name='conv6', activation='relu', use_gpu=use_gpu)
        self.conv7 = layers.conv2d(filters=self.conv3_256_features, kernel=self.conv3_256_kernel, padding='SAME', name='conv7', activation='relu', use_gpu=use_gpu)
        self.max_pool3 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool3')


        self.conv8 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv8', activation='relu', use_gpu=use_gpu)
        self.conv9 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv9', activation='relu', use_gpu=use_gpu)
        self.conv10 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv10', activation='relu', use_gpu=use_gpu)
        self.max_pool4 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool4')

        self.conv11 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv11', activation='relu', use_gpu=use_gpu)
        self.conv12 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv12', activation='relu', use_gpu=use_gpu)
        self.conv13 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv13', activation='relu', use_gpu=use_gpu)
        self.max_pool5 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool5')


        self.fc1 = layers.dense(out_size=self.fc_4096, activation='relu', name='fc1', use_gpu=use_gpu)
        self.fc2 = layers.dense(out_size=self.fc_4096, activation='relu', name='fc2', use_gpu=use_gpu)
        self.fc3 = layers.dense(out_size=self.target_size, activation='relu', name='fc3', use_gpu=use_gpu)
    def save_weight(self, name, layer):
        # print_2txt('process {} completed!'.format(name))
        # np.savez(name, w = layer.w.array, b = layer.b.array)
        pass

    def forward(self, out):
        self.generations += 1
        ts1 = time.time()
        out = self.conv1.forward(out)
        ts2=time.time()
        self.print_2txt("conv1 forward time cost: {}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        # self.save_weight("init_wb/conv1.npz", self.conv1)
        out = self.conv2.forward(out)
        ts2=time.time()
        self.print_2txt("conv2 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        # self.save_weight("init_wb/conv2.npz", self.conv2)
        out = self.max_pool1.forward(out)
        ts2=time.time()
        self.print_2txt("max pool1 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        out = self.conv3.forward(out)
        ts2=time.time()
        self.print_2txt("conv3 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        # self.save_weight("init_wb/conv3.npz", self.conv3)
        out = self.conv4.forward(out)
        ts2=time.time()
        self.print_2txt("conv4 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        # self.save_weight("init_wb/conv4.npz", self.conv4)
        out = self.max_pool2.forward(out)
        ts2=time.time()
        self.print_2txt("max pool2 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()

        out = self.conv5.forward(out)
        ts2=time.time()
        self.print_2txt("conv5 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        # self.save_weight("init_wb/conv5.npz", self.conv5)
        out = self.conv6.forward(out)
        ts2=time.time()
        self.print_2txt("conv6 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        # self.save_weight("init_wb/conv6.npz", self.conv6)
        out = self.conv7.forward(out)
        ts2=time.time()
        self.print_2txt("conv7 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        self.save_weight("init_wb/conv7.npz", self.conv7)
        out = self.max_pool3.forward(out)
        ts2=time.time()
        self.print_2txt("max pool 3 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        out = self.conv8.forward(out)
        ts2=time.time()
        self.print_2txt("conv8 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        # self.save_weight("init_wb/conv8.npz", self.conv8)
        out = self.conv9.forward(out)
        ts2=time.time()
        self.print_2txt("conv9 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        # self.save_weight("init_wb/conv9.npz", self.conv9)
        out = self.conv10.forward(out)
        ts2=time.time()
        self.print_2txt("conv10 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        # self.save_weight("init_wb/conv10.npz", self.conv10)
        out = self.max_pool4.forward(out)
        ts2=time.time()
        self.print_2txt("max pool4 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        out = self.conv11.forward(out)
        ts2=time.time()
        self.print_2txt("conv11 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        # self.save_weight("init_wb/conv11.npz", self.conv11)
        out = self.conv12.forward(out)
        ts2=time.time()
        self.print_2txt("conv12 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        # self.save_weight("init_wb/conv12.npz", self.conv12)
        out = self.conv13.forward(out)
        self.save_weight("init_wb/conv13.npz", self.conv13)
        ts2=time.time()
        self.print_2txt("conv13 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        out = self.max_pool5.forward(out)
        ts2=time.time()
        self.print_2txt("max pool5 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        out = self.fc1.forward(out)
        ts2=time.time()
        self.print_2txt("fc1 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        # self.save_weight("init_wb/fc1.npz", self.fc1)
        out = self.fc2.forward(out)
        ts2=time.time()
        self.print_2txt("fc2 forward time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        # self.save_weight("init_wb/fc2.npz", self.fc2)
        out = self.fc3.forward(out)
        # self.save_weight("init_wb/fc3.npz", self.fc3)
        ts2=time.time()
        self.print_2txt("fc3 forward time cost:{}".format((ts2-ts1)*1000.))
        return out
    def cal_gradients(self, loss, temp_out):
        ts1 = time.time()
        d_out = chainer.grad([loss], [temp_out])
        if isinstance(d_out, (list)):
            d_out = d_out[0]
        d_out = self.fc3.backward(d_out)
        ts2=time.time()
        self.print_2txt("fc3 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.fc2.backward(d_out)
        ts2=time.time()
        self.print_2txt("fc2 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.fc1.backward(d_out)
        ts2=time.time()
        self.print_2txt("fc1 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()

        d_out = self.max_pool5.backward(d_out)
        ts2=time.time()
        self.print_2txt("max pool5 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.conv13.backward(d_out)
        ts2=time.time()
        self.print_2txt("conv13 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.conv12.backward(d_out)
        ts2=time.time()
        self.print_2txt("conv12 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.conv11.backward(d_out)
        ts2=time.time()
        self.print_2txt("conv11 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()

        d_out = self.max_pool4.backward(d_out)
        ts2=time.time()
        self.print_2txt("max pool 4 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.conv10.backward(d_out)
        ts2=time.time()
        self.print_2txt("conv10 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.conv9.backward(d_out)
        ts2=time.time()
        self.print_2txt("conv9 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.conv8.backward(d_out)
        ts2=time.time()
        self.print_2txt("conv8 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.max_pool3.backward(d_out)
        ts2=time.time()
        self.print_2txt("max pool3 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.conv7.backward(d_out)
        ts2=time.time()
        self.print_2txt("conv7 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.conv6.backward(d_out)
        ts2=time.time()
        self.print_2txt("conv6 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.conv5.backward(d_out)
        ts2=time.time()
        self.print_2txt("conv5 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()


        d_out = self.max_pool2.backward(d_out)
        ts2=time.time()
        self.print_2txt("max pool2 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.conv4.backward(d_out)
        ts2=time.time()
        self.print_2txt("conv4 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.conv3.backward(d_out)
        ts2=time.time()
        self.print_2txt("conv3 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()

        d_out = self.max_pool1.backward(d_out)
        ts2=time.time()
        self.print_2txt("max pool1 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.conv2.backward(d_out)
        ts2=time.time()
        self.print_2txt("conv2 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        d_out = self.conv1.backward(d_out)
        ts2=time.time()
        self.print_2txt("conv1 cal_gradients time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        return

    def update(self):
        ts1 = time.time()
        self.conv1.update_parameters()
        ts2=time.time()
        self.print_2txt("conv1 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        self.conv2.update_parameters()
        ts2=time.time()
        self.print_2txt("conv2 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        self.conv3.update_parameters()
        ts2=time.time()
        self.print_2txt("conv3 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        self.conv4.update_parameters()
        ts2=time.time()
        self.print_2txt("conv4 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()

        self.conv5.update_parameters()
        ts2=time.time()
        self.print_2txt("conv5 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        self.conv6.update_parameters()
        ts2=time.time()
        self.print_2txt("conv6 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        self.conv7.update_parameters()
        ts2=time.time()
        self.print_2txt("conv7 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()

        self.conv8.update_parameters()
        ts2=time.time()
        self.print_2txt("conv8 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        self.conv9.update_parameters()
        ts2=time.time()
        self.print_2txt("conv9 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        self.conv10.update_parameters()
        ts2=time.time()
        self.print_2txt("conv10 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()

        self.conv11.update_parameters()
        ts2=time.time()
        self.print_2txt("conv11 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        self.conv12.update_parameters()
        ts2=time.time()
        self.print_2txt("conv12 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        self.conv13.update_parameters()
        ts2=time.time()
        self.print_2txt("conv13 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()

        self.fc1.update_parameters()
        ts2=time.time()
        self.print_2txt("fc1 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        self.fc2.update_parameters()
        ts2=time.time()
        self.print_2txt("fc2 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        self.fc3.update_parameters()
        ts2=time.time()
        self.print_2txt("fc3 update time cost:{}".format((ts2-ts1)*1000.))
        ts1 = time.time()
        return

    def print_2txt(self, s):
        file = open("time_cost.txt", "a+")
        pre = "generations {}".format(self.generations)
        s = pre + s
        file.write(s)
        file.write('\n')
        file.close()
generations = 10
batch_size = 128
use_gpu = True
vgg = VGG16(use_gpu=use_gpu)
file = open("time_cost.txt", "w")
file.close()
for i in range(generations):
    trainX, trainY = utils.get_batch_data(batch_size)
    Y = trainY.astype(np.int32)
    trainX, Y = chainer.as_variable(trainX), chainer.as_variable(Y)
    if use_gpu:
        trainX.to_gpu()
        Y.to_gpu()
    ts1 = time.time()
    temp_out = vgg.forward(trainX)

    loss = F.softmax_cross_entropy(temp_out, Y)
    accuracy = F.accuracy(temp_out, Y)
    print('epoch #{} loss: {}'.format(i, loss))
    print('epoch #{} accuracy: {}'.format(i, accuracy))


    vgg.cal_gradients(loss, temp_out)
    vgg.update()
    ts2 = time.time()
    print("one epoch cost time: ", ts2 - ts1)
    del trainX, trainY, Y
