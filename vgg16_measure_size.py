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
# from model import VGG16
class VGG16():
    batch_size = 256
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

    def __init__(self):
        self.conv1 = layers.conv2d(filters=self.conv3_64_features, kernel=self.conv3_64_kernel, padding='SAME', name='conv1', activation='relu')
        self.conv2 = layers.conv2d(filters=self.conv3_64_features, kernel=self.conv3_64_kernel, padding='SAME', name='conv2', activation='relu')
        self.max_pool1 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool1')

        self.conv3 = layers.conv2d(filters=self.conv3_128_features, kernel=self.conv3_128_kernel, padding='SAME', name='conv3', activation='relu')
        self.conv4 = layers.conv2d(filters=self.conv3_128_features, kernel=self.conv3_128_kernel, padding='SAME', name='conv4', activation='relu')
        self.max_pool2 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool2')

        self.conv5 = layers.conv2d(filters=self.conv3_256_features, kernel=self.conv3_256_kernel, padding='SAME', name='conv5', activation='relu')
        self.conv6 = layers.conv2d(filters=self.conv3_256_features, kernel=self.conv3_256_kernel, padding='SAME', name='conv6', activation='relu')
        self.conv7 = layers.conv2d(filters=self.conv3_256_features, kernel=self.conv3_256_kernel, padding='SAME', name='conv7', activation='relu')
        self.max_pool3 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool3')


        self.conv8 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv8', activation='relu')
        self.conv9 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv9', activation='relu')
        self.conv10 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv10', activation='relu')
        self.max_pool4 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool4')

        self.conv11 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv11', activation='relu')
        self.conv12 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv12', activation='relu')
        self.conv13 = layers.conv2d(filters=self.conv3_512_features, kernel=self.conv3_512_kernel, padding='SAME', name='conv13', activation='relu')
        self.max_pool5 = layers.max_pool2d(ksize=self.max_pool_window, stride=self.max_pool_stride, name='max_pool5')


        self.fc1 = layers.dense(out_size=self.fc_4096, activation='relu', name='fc1')
        self.fc2 = layers.dense(out_size=self.fc_4096, activation='relu', name='fc2')
        self.fc3 = layers.dense(out_size=self.target_size, activation='relu', name='fc3')
    def save_weight(self, name, layer):
        # print('process {} completed!'.format(name))
        # np.savez(name, w = layer.w.array, b = layer.b.array)
        pass
    def save_output(self, name, output):
        print('process {} completed!'.format(name))
        np.savez(name, x=output.array)

    def forward(self, out):
        out = self.conv1.forward(out)
        # self.save_weight("init_wb/conv1.npz", self.conv1)
        self.save_output("conv1_output.npz", out)

        out = self.conv2.forward(out)
        # self.save_weight("init_wb/conv2.npz", self.conv2)
        self.save_output("conv2_output.npz", out)
        out = self.max_pool1.forward(out)

        self.save_output("max_pool1_output.npz", out)
        out = self.conv3.forward(out)
        self.save_output("conv3_output.npz", out)
        # self.save_weight("init_wb/conv3.npz", self.conv3)
        out = self.conv4.forward(out)
        self.save_output("conv4_output.npz", out)
        # self.save_weight("init_wb/conv4.npz", self.conv4)
        out = self.max_pool2.forward(out)

        self.save_output("max_pool2_output.npz", out)

        out = self.conv5.forward(out)
        # self.save_weight("init_wb/conv5.npz", self.conv5)
        self.save_output("conv5_output.npz", out)
        out = self.conv6.forward(out)
        self.save_output("conv6_output.npz", out)
        # self.save_weight("init_wb/conv6.npz", self.conv6)
        out = self.conv7.forward(out)
        self.save_output("conv7_output.npz", out)
        # self.save_weight("init_wb/conv7.npz", self.conv7)
        out = self.max_pool3.forward(out)
        self.save_output("max_pool3_output.npz", out)
        out = self.conv8.forward(out)
        # self.save_weight("init_wb/conv8.npz", self.conv8)
        self.save_output("conv8_output.npz", out)
        out = self.conv9.forward(out)
        # self.save_weight("init_wb/conv9.npz", self.conv9)
        self.save_output("conv9_output.npz", out)
        out = self.conv10.forward(out)
        # self.save_weight("init_wb/conv10.npz", self.conv10)
        self.save_output("conv10_output.npz", out)
        out = self.max_pool4.forward(out)
        self.save_output("max_pool4_output.npz", out)

        out = self.conv11.forward(out)
        self.save_output("conv11_output.npz", out)
        # self.save_weight("init_wb/conv11.npz", self.conv11)
        out = self.conv12.forward(out)
        self.save_output("conv12_output.npz", out)
        # self.save_weight("init_wb/conv12.npz", self.conv12)
        out = self.conv13.forward(out)
        self.save_output("conv13_output.npz", out)
        # self.save_weight("init_wb/conv13.npz", self.conv13)
        out = self.max_pool5.forward(out)
        self.save_output("max_pool5_output.npz", out)
        out = self.fc1.forward(out)
        self.save_output("fc1_output.npz", out)
        # self.save_weight("init_wb/fc1.npz", self.fc1)
        out = self.fc2.forward(out)
        self.save_output("fc2_output.npz", out)
        # self.save_weight("init_wb/fc2.npz", self.fc2)
        out = self.fc3.forward(out)
        self.save_output("fc3_output.npz", out)
        # self.save_weight("init_wb/fc3.npz", self.fc3)
        return out
    def cal_gradients(self, loss, temp_out):
        d_out = chainer.grad([loss], [temp_out])
        if isinstance(d_out, (list)):
            d_out = d_out[0]
        d_out = self.fc3.backward(d_out)
        d_out = self.fc2.backward(d_out)
        d_out = self.fc1.backward(d_out)

        d_out = self.max_pool5.backward(d_out)
        d_out = self.conv13.backward(d_out)
        d_out = self.conv12.backward(d_out)
        d_out = self.conv11.backward(d_out)

        d_out = self.max_pool4.backward(d_out)
        d_out = self.conv10.backward(d_out)
        d_out = self.conv9.backward(d_out)
        d_out = self.conv8.backward(d_out)

        d_out = self.max_pool3.backward(d_out)
        d_out = self.conv7.backward(d_out)
        d_out = self.conv6.backward(d_out)
        d_out = self.conv5.backward(d_out)


        d_out = self.max_pool2.backward(d_out)
        d_out = self.conv4.backward(d_out)
        d_out = self.conv3.backward(d_out)

        d_out = self.max_pool1.backward(d_out)
        d_out = self.conv2.backward(d_out)
        d_out = self.conv1.backward(d_out)
        return

    def update(self):
        self.conv1.update_parameters()
        self.conv2.update_parameters()

        self.conv3.update_parameters()
        self.conv4.update_parameters()

        self.conv5.update_parameters()
        self.conv6.update_parameters()
        self.conv7.update_parameters()

        self.conv8.update_parameters()
        self.conv9.update_parameters()
        self.conv10.update_parameters()

        self.conv11.update_parameters()
        self.conv12.update_parameters()
        self.conv13.update_parameters()

        self.fc1.update_parameters()
        self.fc2.update_parameters()
        self.fc3.update_parameters()
        return

generations = 1
batch_size = 128
vgg = VGG16()
for i in range(generations):
    trainX, trainY = utils.get_batch_data(batch_size)
    Y = trainY.astype(np.int32)
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
