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
import layers
import utils
import gc
import time


class AlexNet():
    insize = 227

    def __init__(self):

        conv_stride = [4, 4]
        self.conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride)
        self.conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1])
        self.conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride)
        self.conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1])
        self.conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1])
        self.conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1])
        self.conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1])

        self.fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6')
        self.fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7')
        self.fc8 = layers.dense(1000, activation='relu', name='fc8')

        self.max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        self.max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        self.max_pool5 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.max_pool1.forward(out)


        out = self.conv2.forward(out)
        out = self.max_pool2.forward(out)



        out = self.conv3.forward(out)


        out = self.conv4.forward(out)

        out = self.conv5.forward(out)

        out = self.max_pool5.forward(out)

        out = self.fc6.forward(out)

        out = self.fc7.forward(out)

        out = self.fc8.forward(out)


        return out

    def backward(self, loss, temp_out):
        d_out = chainer.grad([loss], [temp_out])

        if isinstance(d_out, (list)):
            d_out = d_out[0]
        d_out = self.fc8.backward(d_out)
        

        d_out = self.fc7.backward(d_out)
        d_out = self.fc6.backward(d_out)

        d_out = self.max_pool5.backward(d_out)
        d_out = self.conv5.backward(d_out)

        d_out = self.conv4.backward(d_out)
        d_out = self.conv3.backward(d_out)

        d_out = self.max_pool2.backward(d_out)
        d_out = self.conv2.backward(d_out)

        d_out = self.max_pool1.backward(d_out)
        d_out = self.conv1.backward(d_out)



generations = 1000

batch_size = 128

alexNet = AlexNet()

ts = time.time()
start_time = time.ctime(ts)
print("start time:", start_time)
for i in range(generations):
    trainX, trainY = utils.get_batch_data(batch_size)
    
    Y = trainY.astype(np.int32)
    
    temp_out = alexNet.forward(trainX)

    ts = time.time()
    infer_end_time = time.ctime(ts)
    print("infer_end time:", infer_end_time)

    loss = F.softmax_cross_entropy(temp_out, Y)
    accuracy = F.accuracy(temp_out, Y)
    print('epoch #{} loss: {}'.format(i, loss))
    print('epoch #{} accuracy: {}'.format(i, accuracy))


    alexNet.backward(loss, temp_out)
    ts = time.time()
    end_time = time.ctime(ts)
    print("end time:", end_time)

    del trainX, trainY, Y

