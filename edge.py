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
import args


_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class Connecter(communication_pb2_grpc.CommServicer):

    epoch = 0

    def compute_forward(self, out, process_layers):
        ts1 = time.time()

        if 1 in process_layers:
            out = self.conv1.forward(out)
        if 2 in process_layers:
            out = self.max_pool1.forward(out)
        if 3 in process_layers:
            out = self.conv2.forward(out)
        if 4 in process_layers:
            out = self.max_pool2.forward(out)
        if 5 in process_layers:
            out = self.conv3.forward(out)
        if 6 in process_layers:
            out = self.conv4.forward(out)
        if 7 in process_layers:
            out = self.conv5.forward(out)
        if 8 in process_layers:
            out = self.max_pool5.forward(out)
        if 9 in process_layers:
            out = self.fc6.forward(out)
        if 10 in process_layers:
            out = self.fc7.forward(out)
        if 11 in process_layers:
            out = self.fc8.forward(out)

        ts2 = time.time()
        if self.epoch is not 0:
            if self.epoch == 1:
                self.forward_time =  (ts2 - ts1) * 1000.
            elif self.epoch > 1:
                self.forward_time = ((ts2 - ts1) * 1000.)/self.epoch + self.forward_time * (self.epoch-1)/self.epoch

            print("server forwarding time:", self.forward_time)
        return out

    def compute_backward(self, d_out, Y, process_layers):


        ts1 = time.time()
        if 11 in process_layers:
            loss = F.softmax_cross_entropy(d_out, Y)
            accuracy = F.accuracy(d_out, Y)
            print('loss: {}'.format(loss))
            print('accuracy: {}'.format(accuracy))

            d_out = chainer.grad([loss], [d_out])
            if isinstance(d_out, (list)):
                d_out = d_out[0]
            d_out = fc8.backward(d_out)
        if 10 in process_layers:
            d_out = fc7.backward(d_out)
        if 9 in process_layers:
            d_out = fc6.backward(d_out)
        if 8 in process_layers:
            d_out = max_pool5.backward(d_out)
        if 7 in process_layers:
            d_out = conv5.backward(d_out)
        if 6 in process_layers:
            d_out = conv4.backward(d_out)
        if 5 in process_layers:
            d_out = conv3.backward(d_out)
        if 4 in process_layers:
            d_out = max_pool2.backward(d_out)
        if 3 in process_layers:
            d_out = conv2.backward(d_out)
        if 2 in process_layers:
            d_out = max_pool1.backward(d_out)
        if 1 in process_layers:
            d_out = conv1.backward(d_out)


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

    # initial needed process layers
    def init_layers(self, process_layers):
        conv_stride = [4, 4]

        if 1 in process_layers:
            self.conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride)
        if 2 in process_layers:
            self.max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        if 3 in process_layers:
            self.conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1])
        if 4 in process_layers:
            self.max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        if 5 in process_layers:
            self.conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1])
        if 6 in process_layers:
            self.conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1])
        if 7 in process_layers:
            self.conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1])
        if 8 in process_layers:
            self.max_pool5 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        if 9 in process_layers:
            self.fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6')
        if 10 in process_layers:
            self.fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7')
        if 11 in process_layers:
            self.fc8 = layers.dense(1000, activation='relu', name='fc8')



    #To communicate to device
    def Forwarding(self, request, context):

        process_layers = args.args_prase()
        self.init_layers(process_layers)


        print("Start epoch {} :".format(self.epoch))

        ts1 = time.time()

        input = pickle.loads(request.array)
        Y = pickle.loads(request.Y)

        ts2 = time.time()

        out = self.compute_forward(input, process_layers)

        dout = self.run(out, Y)

        d_out = self.compute_backward(dout, Y, process_layers)

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
