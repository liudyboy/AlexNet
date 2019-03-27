from concurrent import futures
import logging
import grpc

import communication_pb2_grpc
import communication_pb2
import time
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
import numpy as np
import pickle
import sys


_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class Connecter(communication_pb2_grpc.CommServicer):


    conv_stride = [4, 4]
    conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride, use_gpu=True)
    conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1], use_gpu=True)
    conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1], use_gpu=True)
    conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1], use_gpu=True)
    conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1], use_gpu=True)

    fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6', use_gpu=True)
    fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7', use_gpu=True)
    fc8 = layers.dense(1000, activation='relu', name='fc8', use_gpu=True)

    max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
    max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
    max_pool5 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])

    epoch = 0

    def compute_forward(self, input):
        tc2 = time.time()

        input.to_gpu()

        tc3 = time.time()
        out = self.conv1.forward(input)
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

        tc4 = time.time()

        if self.epoch is not 0:
            if self.epoch == 1:
                self.CPU2GPU_time = (tc3 - tc2) * 1000.
                self.forward_time =  (tc4 - tc3) * 1000.
            elif self.epoch > 1:
                self.CPU2GPU_time = ((tc3 - tc2) * 1000.)/self.epoch + self.CPU2GPU_time * (self.epoch-1)/self.epoch
                self.forward_time = ((tc4 - tc3) * 1000.)/self.epoch + self.forward_time * (self.epoch-1)/self.epoch


            print("input data from CPU to GPU, cost time:", self.CPU2GPU_time)
            print("server forwarding time:", self.forward_time)
        return out

    def compute_backward(self, out, Y):

        tc1 = time.time()
        Y.to_gpu()

        loss = F.softmax_cross_entropy(out, Y)
        accuracy = F.accuracy(out, Y)
        print('loss: {}'.format(loss))
        print('accuracy: {}'.format(accuracy))

        d_out = chainer.grad([loss], [out])
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



        tc4 = time.time()
        # d_out.to_cpu()
        d_out = np.ones(shape=(1,1))
        tc5 = time.time()

        if self.epoch is not 0:
            if self.epoch == 1:
                self.backward_time = (tc4 - tc1) * 1000.
                self.GPU2CPU_time = (tc5 - tc4) * 1000.
            elif self.epoch > 1:
                self.backward_time = ((tc4 - tc1) * 1000.)/self.epoch + self.backward_time * (self.epoch-1)/self.epoch
                self.GPU2CPU_time = ((tc5 - tc4) * 1000.)/self.epoch + self.GPU2CPU_time * (self.epoch-1)/self.epoch

            print("server backward time:", self.backward_time)
            print("output data from GPU to CPU, cost time:", self.GPU2CPU_time)
        return d_out

    def Forwarding(self, request, context):
        
        print("Start epoch {} :".format(self.epoch))

        ts1 = time.time()

        input = pickle.loads(request.array)
        Y = pickle.loads(request.Y)

        ts2 = time.time()

        out = self.compute_forward(input)
        d_out = self.compute_backward(out, Y)

        ts5 = time.time()
        d_out = pickle.dumps(d_out)

        self.epoch += 1

        ts3 = time.time()
        print("server cost time:", (ts3 - ts1) * 1000.)

        size = sys.getsizeof(d_out)
        print("send client data size:", (size/1024./1024.))


        if self.epoch is not 0:
            if self.epoch == 1:
                self.change_format_time = (ts2 - ts1 + ts3 -ts5) * 1000.
            elif self.epoch > 1:
                self.change_format_time = ((ts2 - ts1 + ts3 - ts5) * 1000.)/self.epoch + self.change_format_time * (self.epoch-1)/self.epoch

            print("change received data format, cost time:", self.change_format_time)

        return communication_pb2.ArrayReply(array=d_out)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)])
    communication_pb2_grpc.add_CommServicer_to_server(Connecter(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    logging.basicConfig()
    serve()
