from rpyc import Service
from rpyc.utils.server import ThreadedServer
from threading import Thread
import rpyc
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
import numpy as np

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

class Server(Service):

    fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7')
    fc8 = layers.dense(1000, activation='relu', name='fc8')

    def exposed_forward(self, input, Y):
        input = np.asarray(input)
        Y = np.asarray(Y)
        # print('input type: ', type(input))
        # print('Y type: ', type(Y))
        Y.astype(np.int32)

        input = chainer.as_variable(input)
        out = self.fc7.forward(input)
        out = self.fc8.forward(out)

        loss = F.softmax_cross_entropy(out, Y)
        accuracy = F.accuracy(out, Y)
        print('epoch  loss: {}'.format(loss))
        print('epoch  accuracy: {}'.format(accuracy))

        d_out = chainer.grad([loss], [out])
        if isinstance(d_out, (list)):
            d_out = d_out[0]
        d_out = self.fc8.backward(d_out)
        d_out = self.fc7.backward(d_out)
        return d_out.array





if __name__ == '__main__':
    myserver= Server()
    s = ThreadedServer(myserver, port=18871, protocol_config=rpyc.core.protocol.DEFAULT_CONFIG)

    s.start()
