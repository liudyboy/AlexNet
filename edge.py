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
import layers
import utils
import gc
import time
import numpy as np
import pickle
import sys


_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class Connecter(communication_pb2_grpc.CommServicer):

    epoch = 0
    conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1])
    # conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1])
    # conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1])
    # conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1])

    # fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6')
    # fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7')
    # fc8 = layers.dense(1000, activation='relu', name='fc8')

    # max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
    max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])

    def compute_forward(self, input):
        ts1 = time.time()
        out = self.conv2.forward(input)
        out = self.max_pool2.forward(out)

        ts2 = time.time()
        if self.epoch is not 0:
            if self.epoch == 1:
                self.forward_time =  (ts2 - ts1) * 1000.
            elif self.epoch > 1:
                self.forward_time = ((ts2 - ts1) * 1000.)/self.epoch + self.forward_time * (self.epoch-1)/self.epoch

            print("server forwarding time:", self.forward_time)
        return out

    def compute_backward(self, out, Y):
        ts1 = time.time()
        d_out = self.max_pool2.backward(out)
        d_out = self.conv2.backward(d_out)
        ts2 = time.time()
        if self.epoch is not 0:
            if self.epoch == 1:
                self.backward_time = (ts2 - ts1) * 1000.
            elif self.epoch > 1:
                self.backward_time = ((ts2 - ts1) * 1000.)/self.epoch + self.backward_time * (self.epoch-1)/self.epoch

            print("server backward time:", self.backward_time)
        return d_out

    #To communicate to cloud
    def run(self, sendArray, Y):
        with grpc.insecure_channel("192.168.1.153:50052", options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)]) as channel:
            stub = communication_pb2_grpc.CommStub(channel)
            sendArray = pickle.dumps(sendArray)
            Y = pickle.dumps(Y)
            recv_array = stub.Forwarding(communication_pb2.ArrayRecv(array=sendArray, Y=Y))
            recv_array = pickle.loads(recv_array.array)
        return recv_array


    #To communicate to device
    def Forwarding(self, request, context):
        print("Start epoch {} :".format(self.epoch))

        ts1 = time.time()

        input = pickle.loads(request.array)
        Y = pickle.loads(request.Y)

        ts2 = time.time()

        out = self.compute_forward(input)

        dout = self.run(out, Y)

        d_out = self.compute_backward(dout, Y)

        if self.epoch is not 0:
            if self.epoch == 1:
                self.change_format_time = (ts2 - ts1) * 1000.
            elif self.epoch > 1:
                self.change_format_time = ((ts2 - ts1) * 1000.)/self.epoch + self.change_format_time * (self.epoch-1)/self.epoch

            print("change received data to chainer, cost time:", self.change_format_time)

        d_out = pickle.dumps(d_out)

        self.epoch += 1

        ts3 = time.time()
        print("server cost time:", (ts3 - ts1) * 1000.)

        size = sys.getsizeof(d_out)
        print("send client data size:", (size/1024./1024.))



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
