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


class AlexNet():
    insize = 227

    def __init__(self):

        conv_stride = [4, 4]
        self.conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride)
        self.conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1])
        self.conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1])
        self.conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1])
        self.conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1])

        self.fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6')
        self.fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7')
        self.fc8 = layers.dense(200, activation='relu', name='fc8')

        self.max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        self.max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        self.max_pool5 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])

    def forward(self, x):

        ts1 = time.time()

        out = self.conv1.forward(x)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("time_cost.txt", 'a')
        file.write('layer 1 forward time:  {} \n'.format(cost))
        file.close()


        ts1 = time.time()

        out = self.max_pool1.forward(out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("time_cost.txt", 'a')
        file.write('layer 2 forward time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        out = self.conv2.forward(out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("time_cost.txt", 'a')
        file.write('layer 3 forward time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        out = self.max_pool2.forward(out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("time_cost.txt", 'a')
        file.write('layer 4 forward time:  {} \n'.format(cost))
        file.close()


        ts1 = time.time()

        out = self.conv3.forward(out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("time_cost.txt", 'a')
        file.write('layer 5 forward time:  {} \n'.format(cost))
        file.close()


        ts1 = time.time()

        out = self.conv4.forward(out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("time_cost.txt", 'a')
        file.write('layer 6 forward time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        out = self.conv5.forward(out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("time_cost.txt", 'a')
        file.write('layer 7 forward time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        out = self.max_pool5.forward(out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("time_cost.txt", 'a')
        file.write('layer 8 forward time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        out = self.fc6.forward(out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("time_cost.txt", 'a')
        file.write('layer 9 forward time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        out = self.fc7.forward(out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("time_cost.txt", 'a')
        file.write('layer 10 forward time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        out = self.fc8.forward(out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("time_cost.txt", 'a')
        file.write('layer 11 forward time:  {} \n'.format(cost))
        file.close()

        return out

    def backward(self, loss, temp_out):

        ts1 = time.time()

        d_out = chainer.grad([loss], [temp_out])

        if isinstance(d_out, (list)):
            d_out = d_out[0]
        d_out = self.fc8.backward(d_out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("gradient_cost.txt", 'a')
        file.write('layer 11 cal gradients time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        d_out = self.fc7.backward(d_out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("gradient_cost.txt", 'a')
        file.write('layer 10 cal gradients time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        d_out = self.fc6.backward(d_out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("gradient_cost.txt", 'a')
        file.write('layer 9 cal gradients time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        d_out = self.max_pool5.backward(d_out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("gradient_cost.txt", 'a')
        file.write('layer 8 cal gradients time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        d_out = self.conv5.backward(d_out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("gradient_cost.txt", 'a')
        file.write('layer 7 cal gradients time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        d_out = self.conv4.backward(d_out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("gradient_cost.txt", 'a')
        file.write('layer 6 cal gradients time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        d_out = self.conv3.backward(d_out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("gradient_cost.txt", 'a')
        file.write('layer 5 cal gradients time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        d_out = self.max_pool2.backward(d_out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("gradient_cost.txt", 'a')
        file.write('layer 4 cal gradients time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        d_out = self.conv2.backward(d_out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("gradient_cost.txt", 'a')
        file.write('layer 3 cal gradients time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        d_out = self.max_pool1.backward(d_out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("gradient_cost.txt", 'a')
        file.write('layer 2 cal gradients time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        d_out = self.conv1.backward(d_out)

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("gradient_cost.txt", 'a')
        file.write('layer 1 cal gradients time:  {} \n'.format(cost))
        file.close()

    def update(self):
        ts1 = time.time()

        self.conv1.update_parameters()

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("update_cost.txt", 'a')
        file.write('layer 1 update time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        self.conv2.update_parameters()

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("update_cost.txt", 'a')
        file.write('layer 3 update time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        self.conv3.update_parameters()

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("update_cost.txt", 'a')
        file.write('layer 5 update time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        self.conv4.update_parameters()

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("update_cost.txt", 'a')
        file.write('layer 6 update time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        self.conv5.update_parameters()

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("update_cost.txt", 'a')
        file.write('layer 7 update time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        self.fc6.update_parameters()

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("update_cost.txt", 'a')
        file.write('layer 9 update time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        self.fc7.update_parameters()

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("update_cost.txt", 'a')
        file.write('layer 10  update time:  {} \n'.format(cost))
        file.close()

        ts1 = time.time()

        self.fc8.update_parameters()

        ts2 = time.time()
        cost = (ts2-ts1) * 1000.
        file = open("update_cost.txt", 'a')
        file.write('layer 11 update time:  {} \n'.format(cost))
        file.close()


ts1 = time.time()
start_time = time.ctime(ts1)
generations = 10

batch_size = 128

alexNet = AlexNet()

file = open('time_cost.txt', 'w')
file.close()
file = open('gradient_cost.txt', 'w')
file.close()
file = open('update_cost.txt', 'w')
file.close()

for i in range(generations):

    trainX, trainY = utils.get_batch_data(batch_size)
    Y = trainY.astype(np.int32)

    ts1 = time.time()
    temp_out = alexNet.forward(trainX)

    loss = F.softmax_cross_entropy(temp_out, Y)
    accuracy = F.accuracy(temp_out, Y)
    print('epoch #{} loss: {}'.format(i, loss))
    print('epoch #{} accuracy: {}'.format(i, accuracy))


    alexNet.backward(loss, temp_out)
    alexNet.update()
    del trainX, trainY, Y

    ts2 = time.time()

    print("one epoch process time" , (ts2 - ts1) * 1000.)
