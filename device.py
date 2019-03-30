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
import utils
import gc
import time
import sys


conv_stride = [4, 4]
conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride)
# conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1])
# conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1])
# conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1])
# conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1])

# fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6')
# fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7')
# fc8 = layers.dense(1000, activation='relu', name='fc8')

max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
# max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
# max_pool5 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])

def Forward(x):
    out = conv1.forward(x)
    out = max_pool1.forward(out)
    
    
    # out = conv2.forward(out)
    # out = max_pool2.forward(out)
    
    

    # out = conv3.forward(out)
    
    
    # out = conv4.forward(out)
    
    # out = conv5.forward(out)

    # out = max_pool5.forward(out)

    # out = fc6.forward(out)

    # out = fc7.forward(out)

    # out = fc8.forward(out)


    return out

def Backward(d_out):

    # d_out = chainer.grad([loss], [temp_out])
    
    # if isinstance(d_out, (list)):
    #     d_out = d_out[0]
    # d_out = fc8.backward(d_out)
        
        
    # d_out = fc7.backward(d_out)


    # d_out = fc6.backward(d_out)
    

    # d_out = max_pool5.backward(d_out)


    # d_out = conv5.backward(d_out)


    # d_out = conv4.backward(d_out)


    # d_out = conv3.backward(d_out)

    # d_out = max_pool2.backward(d_out)



    # d_out = conv2.backward(d_out)

    d_out = max_pool1.backward(d_out)



    d_out = conv1.backward(d_out)

    del d_out



def run(sendArray, Y):
    with grpc.insecure_channel("192.168.1.77:50051", options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)]) as channel:
        stub = communication_pb2_grpc.CommStub(channel)
        sendArray = pickle.dumps(sendArray)
        Y = pickle.dumps(Y)
        recv_array = stub.Forwarding(communication_pb2.ArrayRecv(array=sendArray, Y=Y))
        recv_array = pickle.loads(recv_array.array)
    return recv_array

if __name__ == "__main__":
    logging.basicConfig()
    ts = time.time()
    start_time = time.ctime(ts)
    print("start time:", start_time)
    generations = 100
    batch_size = 64


    for i in range(generations):

        ts1 = time.time()

        trainX, trainY = utils.get_batch_data(batch_size)
        trainX = chainer.as_variable(trainX)
        Y = trainY.astype(np.int32)
        Y = chainer.as_variable(Y)

        output = Forward(trainX)

        ts3 = time.time()
        dout = run(output, Y)

        ts4 = time.time()
        Backward(dout)

        ts2 = time.time()
        process_time = ts2 - ts1
        client_compute_time = ts3-ts1 + ts2-ts4

        if i is not 0:
            if i == 1:
                complete_time = process_time*1000.
                client_used_time = client_compute_time*1000.
                server_used_time = (complete_time - client_used_time)
            elif i > 1:
                complete_time = (process_time*1000.)/i + complete_time*(i-1)/i
                client_used_time = (client_compute_time*1000.)/i + client_used_time*(i-1)/i
                server_used_time = complete_time - client_used_time

            print("#epoch {} completed!  Used time {}".format(i, complete_time))
            print("client computing time: ", client_used_time)
            print("server cost time: ", server_used_time)

        del trainX, trainY, Y
