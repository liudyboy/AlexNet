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
import connect

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class Connecter(communication_pb2_grpc.CommServicer):
    addressB = '192.168.1.77:50055'
    recv_raw_time = 0
    recv_output_time = 0
    count = 0
    def send_raw_data(self, destination, X, Y):
        singal = connect.conn_send_raw_data(destination, X.array, Y.array)
        print("singal shape: ", singal.shape)

    def process_device_output(self, request, context):
        self.device_output_x = chainer.as_variable(pickle.loads(request.output))
        self.device_output_y = chainer.as_variable(pickle.loads(request.Y))
        self.device_output_flag = True
        self.recv_output_time = time.time()
        if self.recv_raw_time != 0:
            print("raw ahead output time: ", (self.recv_output_time - self.recv_raw_time) * 1000)
        self.count += 1

        # if self.count != 2:
        #     pass
        grads = pickle.dumps(np.arange(1000*1000))
        return communication_pb2.OutputReply(grads=grads)

    # Function call by device
    # Parames: raw_data_x, layer_y
    def process_raw_data(self, request, context):

        raw_input = chainer.as_variable(pickle.loads(request.raw_x))
        self.Y = chainer.as_variable(pickle.loads(request.Y))
        print("get data form A")
        data = chainer.as_variable(np.array([1]))
        y = chainer.as_variable(np.array([3]))
        self.recv_raw_time = time.time()
        if self.recv_output_time != 0:
            print("outptu ahead raw time: ", (self.recv_raw_time - self.recv_output_time) * 1000)
        # ts1 = time.time()
        # self.send_raw_data(self.addressB, data, y)
        # ts2 = time.time()
        # print("receive data from B cost time:", (ts2 - ts1) * 1000)

        finsh_signal = pickle.dumps(np.zeros(1))
        return communication_pb2.RawReply(signal=finsh_signal)

    def Log(self, message):
        # print(message)
        pass
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
