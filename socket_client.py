import numpy as np
import sys
import pickle, struct
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
import layers
import utils
import gc
import time

conv_stride = [4, 4]
conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride)
conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1])
conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride)
conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1])
conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1])
conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1])
conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1])

fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6')
# fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7')
# fc8 = layers.dense(1000, activation='relu', name='fc8')

max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
max_pool5 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])

def Forward(x):
    out = conv1.forward(x)
    out = max_pool1.forward(out)
    
    
    out = conv2.forward(out)
    out = max_pool2.forward(out)
    
    

    out = conv3.forward(out)
    
    
    out = conv4.forward(out)
    
    out = conv5.forward(out)

    out = max_pool5.forward(out)

    out = fc6.forward(out)

    # out = fc7.forward(out)

    # out = fc8.forward(out)


    return out

def Backward(d_out):

    # d_out = chainer.grad([loss], [temp_out])
    
    # if isinstance(d_out, (list)):
    #     d_out = d_out[0]
    # d_out = fc8.backward(d_out)
        
        
    # d_out = fc7.backward(d_out)


    d_out = fc6.backward(d_out)
    

    d_out = max_pool5.backward(d_out)


    d_out = conv5.backward(d_out)


    d_out = conv4.backward(d_out)


    d_out = conv3.backward(d_out)

    d_out = max_pool2.backward(d_out)



    d_out = conv2.backward(d_out)

    d_out = max_pool1.backward(d_out)



    d_out = conv1.backward(d_out)

    del d_out


address_info = ('192.168.1.153', 25000)
def send_from(arr, dest):
    view = memoryview(arr).cast('B')
    while len(view):
        nsent = dest.send(view)
        view = view[nsent:]
def recv_into(arr, source):
    view = memoryview(arr).cast('B')
    while len(view):
        nrecv = source.recv_into(view)
        view = view[nrecv:]

if __name__ == "__main__":

    conn = socket(AF_INET, SOCK_STREAM)

    generations = 1
    batch_size = 128


    for i in range(generations):

        ts1 = time.time()

        trainX, trainY = utils.get_batch_data(batch_size)

        Y = trainY.astype(np.int32)

        print('trainX shape:', trainX.shape)
        fc6_out = Forward(trainX)

        conn.connect(address_info)
        print("client send array shape", fc6_out.array.shape)
        print("client send Y shape", Y.shape)

        send_from(fc6_out.array, conn)
        # send_from(Y, conn)

        print("send data end!")
        # fc7_dout = conn.root.forward(fc6_out.array, Y)

        # fc7_dout = np.asarray(fc7_dout)
        # fc7_dout = chainer.as_variable(fc7_dout)

        # Backward(fc7_dout)

        # print("#epoch {} completed!  Used time {}".format(i, process_time*1000.))
        # print("client computing time: ", client_compute_time*1000.)
        # print("transfer data time: ", (process_time - client_compute_time)*1000.)
        conn.close()


        del trainX, trainY, Y

