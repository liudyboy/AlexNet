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
import gc
import time
import numpy as np
import pickle
import sys
import args
import connect
from model import AlexNet


_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class Connecter(communication_pb2_grpc.CommServicer):

    init_layers_flag = False
    def init_variables(self):
        self.TOTAL_BATCH = 128
        self.edge_address = '192.168.1.70:50055'


    def get_one_layer_gradients(self, destination, grads_w, grads_bias, layer_num):
        grads_w = chainer.as_variable(grads_w)
        grads_bias = chainer.as_variable(grads_bias)
        recv_grads_w, recv_grads_bias = connect.conn_get_gradients(destination, grads_w.array, grads_bias.array, layer_num, 'cloud')
        return chainer.as_variable(recv_grads_w), chainer.as_variable(recv_grads_bias)


    def process_gradients_exchange(self, layer_num):
        edge_address = self.edge_address
        destination = edge_address
        max_pool_layer = [2, 4, 8]
        if layer_num not in max_pool_layer:
            grads_w, grads_bias = self.alexnet.get_params_grads(layer_num)
            grads_w, grads_bias = self.get_one_layer_gradients(destination, grads_w, grads_bias, layer_num)
            self.alexnet.add_params_grads(layer_num, grads_w, grads_bias)
        return

    def send_output_data(self, destination, output, Y):
        ts1 = time.time()
        print('Send output to Cloud start time: ', ts1)
        reply = connect.conn_send_cloud_output_data(destination, output.array, Y.array)
        reply = chainer.as_variable(reply)
        ts2 = time.time()
        print("edge send output to cloud cost time: ", (ts2 - ts1) * 1000.)
        return reply

    # Function call by device
    # Params: raw_data_x, layer_y
    def process_raw_data(self, request, context):
        if self.init_layers_flag is False:
            self.Log('Initial Model layers')
            my_args = args.args_prase()
            self.device_run_layers, self.cloud_run_layers = my_args.M1, my_args.M2
            process_layers = np.arange(1, self.cloud_run_layers+1)
            self.alexnet = AlexNet()
            self.alexnet.init_layers(process_layers)
        self.init_variables()

        ts1 = time.time()
        print("Get raw data time: ", ts1)
        self.raw_input = chainer.as_variable(pickle.loads(request.raw_x))
        self.Y = chainer.as_variable(pickle.loads(request.Y))
        process_layers = np.arange(1, self.cloud_run_layers+1)
        output = self.alexnet.forward(self.raw_input, process_layers)

        output_reply = self.send_output_data(self.edge_address, output, self.Y)

        self.alexnet.cal_gradients(output_reply, process_layers, self.Y)

        for j in process_layers:
            ts1 = time.time()
            self.process_gradients_exchange(j)
            self.alexnet.update_one_layer_parameters(j, self.TOTAL_BATCH)
            ts2 = time.time()
            self.Log('update {} layer cost time: {}'.format(j, (ts2 - ts1) * 1000.))


        finsh_signal = pickle.dumps(np.zeros(1))
        return communication_pb2.RawReply(signal=finsh_signal)
    def Log(self, message):
        print(message)
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
