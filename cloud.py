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
import layersM_gpu as layers
import tiny_utils as utils
import gc
import time
import numpy as np
import pickle
import sys
import args


_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class Connecter(communication_pb2_grpc.CommServicer):

    epoch = 0
    init_flag = False
    Data_paralle_init_flag = False
    def compute_forward(self, out, process_layers):
        ts1 = time.time()

        out.to_gpu()

        ts3 = time.time()

        print("out from CPU to GPU:", (ts3 - ts1) * 1000.)
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

            print("cloud forwarding time:", self.forward_time)
        return out

    def compute_backward(self, d_out, Y, process_layers):

        ts1 = time.time()
        Y.to_gpu()
        if 11 in process_layers:
            loss = F.softmax_cross_entropy(d_out, Y)
            accuracy = F.accuracy(d_out, Y)
            print('loss: {}'.format(loss))
            print('accuracy: {}'.format(accuracy))

            d_out = chainer.grad([loss], [d_out])
            if isinstance(d_out, (list)):
                d_out = d_out[0]
            d_out = self.fc8.backward(d_out)
        if 10 in process_layers:
            d_out = self.fc7.backward(d_out)
        if 9 in process_layers:
            d_out = self.fc6.backward(d_out)
        if 8 in process_layers:
            d_out = self.max_pool5.backward(d_out)
        if 7 in process_layers:
            d_out = self.conv5.backward(d_out)
        if 6 in process_layers:
            d_out = self.conv4.backward(d_out)
        if 5 in process_layers:
            d_out = self.conv3.backward(d_out)
        if 4 in process_layers:
            d_out = self.max_pool2.backward(d_out)
        if 3 in process_layers:
            d_out = self.conv2.backward(d_out)
        if 2 in process_layers:
            d_out = self.max_pool1.backward(d_out)
        if 1 in process_layers:
            d_out = self.conv1.backward(d_out)

        d_out.to_cpu()

        ts2 = time.time()
        if self.epoch is not 0:
            if self.epoch == 1:
                self.backward_time = (ts2 - ts1) * 1000.
            elif self.epoch > 1:
                self.backward_time = ((ts2 - ts1) * 1000.)/self.epoch + self.backward_time * (self.epoch-1)/self.epoch

            print("cloud backward time:", self.backward_time)
        return d_out

    def init_layers(self, process_layers):
        self.init_flag = True
        conv_stride = [4, 4]
        if 1 in process_layers:
            self.conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride, use_gpu=True, paralle=paralle)
            c = np.load("init_wb/conv1.npz")
            self.conv1.w, self.conv1.b = c['w'], c['b']
            self.conv1.w, self.conv1.b = chainer.as_variable(self.conv1.w), chainer.as_variable(self.conv1.b)
        if 2 in process_layers:
            self.max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        if 3 in process_layers:
            self.conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1], use_gpu=True, paralle=paralle)
            c = np.load("init_wb/conv2.npz")
            self.conv2.w, self.conv2.b = c['w'], c['b']
            self.conv2.w, self.conv2.b = chainer.as_variable(self.conv2.w), chainer.as_variable(self.conv2.b)
        if 4 in process_layers:
            self.max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        if 5 in process_layers:
            self.conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1], use_gpu=True, paralle=paralle)
            c = np.load("init_wb/conv3.npz")
            self.conv3.w, self.conv3.b = c['w'], c['b']
            self.conv3.w, self.conv3.b = chainer.as_variable(self.conv3.w), chainer.as_variable(self.conv3.b)
        if 6 in process_layers:
            self.conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1], use_gpu=True, paralle=paralle)
            c = np.load("init_wb/conv4.npz")
            self.conv4.w, self.conv4.b = c['w'], c['b']
            self.conv4.w, self.conv4.b = chainer.as_variable(self.conv4.w), chainer.as_variable(self.conv4.b)
        if 7 in process_layers:
            self.conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1], use_gpu=True, paralle=paralle)
            c = np.load("init_wb/conv5.npz")
            self.conv5.w, self.conv5.b = c['w'], c['b']
            self.conv5.w, self.conv5.b = chainer.as_variable(self.conv5.w), chainer.as_variable(self.conv5.b)
        if 8 in process_layers:
            self.max_pool5 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        if 9 in process_layers:
            self.fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6', use_gpu=True, paralle=paralle)
            c = np.load("init_wb/fc6.npz")
            self.fc6.w, self.fc6.b = c['w'], c['b']
            self.fc6.w, self.fc6.b = chainer.as_variable(self.fc6.w), chainer.as_variable(self.fc6.b)
        if 10 in process_layers:
            self.fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7', use_gpu=True, paralle=paralle)
            c = np.load("init_wb/fc7.npz")
            self.fc7.w, self.fc7.b = c['w'], c['b']
            self.fc7.w, self.fc7.b = chainer.as_variable(self.fc7.w), chainer.as_variable(self.fc7.b)
        if 11 in process_layers:
            self.fc8 = layers.dense(200, activation='relu', name='fc8', use_gpu=True, paralle=paralle)
            c = np.load("init_wb/fc8.npz")
            self.fc8.w, self.fc8.b = c['w'], c['b']
            self.fc8.w, self.fc8.b = chainer.as_variable(self.fc8.w), chainer.as_variable(self.fc8.b)

    #To communicate to edge
    def Forwarding(self, request, context):
        if self.init_flag is False:
            self.process_layers = args.args_prase()
            self.init_layers(self.process_layers)

        print("Start epoch {} :".format(self.epoch))
        ts1 = time.time()

        input = pickle.loads(request.array)
        Y = pickle.loads(request.Y)
        input = chainer.as_variable(input)

        out = self.compute_forward(input, self.process_layers)

        d_out = self.compute_backward(out, Y, self.process_layers)

        if 1 in self.process_layers:
            d_out = np.zeros(shape=(1,1))
        d_out = pickle.dumps(d_out)

        self.epoch += 1

        ts3 = time.time()
        print("cloud cost time:", (ts3 - ts1) * 1000.)

        size = sys.getsizeof(d_out)
        print("send edge data size:", (size/1024./1024.))

        return communication_pb2.ArrayReply(array=d_out)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)])
    communication_pb2_grpc.add_CommServicer_to_server(Connecter(), server)
    server.add_insecure_port("[::]:50052")
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    logging.basicConfig()
    serve()
