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
import layersM as layers
import tiny_utils as utils
import gc
import time
import numpy as np
import pickle
import sys
import args


_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class Connecter(communication_pb2_grpc.CommServicer):
    init_layers_flag = False
    # initial needed process layers
    def init_layers(self, process_layers):
        self.init_layers_flag = True
        conv_stride = [4, 4]

        if 1 in process_layers:
            self.conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride)
            c = np.load("init_wb/conv1.npz")
            self.conv1.w, self.conv1.b = c['w'], c['b']
            self.conv1.w, self.conv1.b = chainer.as_variable(self.conv1.w), chainer.as_variable(self.conv1.b)
        if 2 in process_layers:
            self.max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        if 3 in process_layers:
            self.conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1])
            c = np.load("init_wb/conv2.npz")
            self.conv2.w, self.conv2.b = c['w'], c['b']
            self.conv2.w, self.conv2.b = chainer.as_variable(self.conv2.w), chainer.as_variable(self.conv2.b)
        if 4 in process_layers:
            self.max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        if 5 in process_layers:
            self.conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1])
            c = np.load("init_wb/conv3.npz")
            self.conv3.w, self.conv3.b = c['w'], c['b']
            self.conv3.w, self.conv3.b = chainer.as_variable(self.conv3.w), chainer.as_variable(self.conv3.b)
        if 6 in process_layers:
            self.conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1])
            c = np.load("init_wb/conv4.npz")
            self.conv4.w, self.conv4.b = c['w'], c['b']
            self.conv4.w, self.conv4.b = chainer.as_variable(self.conv4.w), chainer.as_variable(self.conv4.b)
        if 7 in process_layers:
            self.conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1])
            c = np.load("init_wb/conv5.npz")
            self.conv5.w, self.conv5.b = c['w'], c['b']
            self.conv5.w, self.conv5.b = chainer.as_variable(self.conv5.w), chainer.as_variable(self.conv5.b)
        if 8 in process_layers:
            self.max_pool5 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        if 9 in process_layers:
            self.fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6')
            c = np.load("init_wb/fc6.npz")
            self.fc6.w, self.fc6.b = c['w'], c['b']
            self.fc6.w, self.fc6.b = chainer.as_variable(self.fc6.w), chainer.as_variable(self.fc6.b)
        if 10 in process_layers:
            self.fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7')
            c = np.load("init_wb/fc7.npz")
            self.fc7.w, self.fc7.b = c['w'], c['b']
            self.fc7.w, self.fc7.b = chainer.as_variable(self.fc7.w), chainer.as_variable(self.fc7.b)
        if 11 in process_layers:
            self.fc8 = layers.dense(200, activation='relu', name='fc8')
            c = np.load("init_wb/fc8.npz")
            self.fc8.w, self.fc8.b = c['w'], c['b']
            self.fc8.w, self.fc8.b = chainer.as_variable(self.fc8.w), chainer.as_variable(self.fc8.b)

    def cal_forward(self, out, process_layers):
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
        return out

    def cal_gradients(self, d_out, process_layers, Y=None):
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

        return d_out

    def update_layers_parameters(self, process_layers, batch_size):
        if 1 in process_layers:
            self.conv1.update_parameters(batch=batch_size)
        if 2 in process_layers:
            pass          # it is maxpool layer needn't exchange gradients
        if 3 in process_layers:
            self.conv2.update_parameters(batch=batch_size)
        if 4 in process_layers:
            pass          # it is maxpool layer needn't exchange gradients
        if 5 in process_layers:
            self.conv3.update_parameters(batch=batch_size)
        if 6 in process_layers:
            self.conv4.update_parameters(batch=batch_size)
        if 7 in process_layers:
            self.conv5.update_parameters(batch=batch_size)
        if 8 in process_layers:
            pass          # it is maxpool layer needn't exchange gradients
        if 9 in process_layers:
            self.fc6.update_parameters(batch=batch_size)
        if 10 in process_layers:
            self.fc7.update_parameters(batch=batch_size)
        if 11 in process_layers:
            self.fc8.update_parameters(batch=batch_size)
        return


    # Function call by device
    # Parames: raw_data_x, layer_y
    def process_raw_data(self, request, context):
        if self.init_layers_flag is False:
            print('initial the Model Layers')
            edge_run_layers = 11
            process_layers = np.arange(edge_run_layers+1)
            self.init_layers(process_layers)

        raw_input = chainer.as_variable(pickle.loads(request.raw_x))
        self.Y = chainer.as_variable(pickle.loads(request.Y))
        process_layers = np.arange(12)
        batch_size = raw_input.shape[0]
        out = self.cal_forward(raw_input, process_layers)
        out = self.cal_gradients(out, process_layers, self.Y)
        self.update_layers_parameters(process_layers, batch_size)

        finsh_signal = pickle.dumps(np.zeros(1))
        return communication_pb2.RawReply(signal=finsh_signal)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)])
    communication_pb2_grpc.add_CommServicer_to_server(Connecter(), server)
    server.add_insecure_port("[::]:50055")
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    logging.basicConfig()
    serve()
