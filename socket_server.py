import socketserver
import pickle
import struct
import sys
import numpy as np
from socket import *
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
import layers_gpu as layers
import utils
import gc
import time

fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7', use_gpu=True)
fc8 = layers.dense(1000, activation='relu', name='fc8', use_gpu=True)


# address_info = ('192.168.1.153', 9999)
address_info = ('', 25000)
def send_from(arr, dest):
    view = memoryview(arr).cast('B')
    while len(view):
        nsent = dest.send(view)
        view = view[nsent:]
def recv_into(arr, source):
    view = memoryview(arr).cast('B')
    while len(view):
        print("view lengh: ", len(view))
        nrecv = source.recv_into(view)
        print("nrecv: ", nrecv)
        view = view[nrecv:]



if __name__ == "__main__":
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind(address_info)
    sock.listen(1000)

    while True:
        conn, address = sock.accept()
        print("conned machine address:", address)
        fc6_out = np.zeros(shape=(128, 3, 227, 227), dtype=np.float32)
        Y = np.zeros(shape=(128,), dtype=np.int32)
        recv_into(fc6_out, conn)
        # recv_into(Y, conn)

        print("data received!")
        input = fc6_out
        Y = chainer.as_variable(Y)
        input = chainer.as_variable(input)
        input.to_gpu()
        Y.to_gpu()
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
        d_out.to_cpu()

        print('d_out shape', d_out.array.shape)

        tc2 = time.time()
        cost_time = tc2 - tc1
        print("cost time: ", cost_time*1000.)
        conn.close()
