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

    target = 0
    cloud_address = "192.168.1.70:50055"
    def send_output_data(self,  destination, output, Y):
        print('Send output to cloud')
        ts1 = time.time()
        reply = connect.conn_send_device_output_data(destination, output.array, Y.array)
        ts2 = time.time()
        print('send output to cloud cost time: ', (ts2 - ts1) * 1000.)

    # Function call by device
    # Parames: raw_data_x, layer_y
    def process_raw_data(self, request, context):
        self.target += 1
        raw_input = chainer.as_variable(pickle.loads(request.raw_x))
        Y = chainer.as_variable(pickle.loads(request.Y))

        data = np.array([1])
        self.send_output_data(self.cloud_address, chainer.as_variable(data), Y)
        finsh_signal = pickle.dumps(data)

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
