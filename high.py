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
    ready_for_new_epoch = False
    finished_epoch = False
    def init_variables(self):
        self.TOTAL_BATCH_SZIE = 128
        self.device_output_flag = False
        self.device_output_gradients_flag = False
        self.device_output_grads = None
        self.cloud_output_flag = False
        self.cloud_output_gradients_flag = False
        self.cloud_output_grads = None
        self.layers_gradients_flag = np.zeros(12)
        self.layers_gradients_flag[0] = 2            # for target[0] initial value is 1
        self.layers_gradients_flag_target = np.zeros(12)
        self.prepared_for_recv_gradients = False

        self.ready_for_new_epoch = True # assure that new epoch is start by devcie call process_raw_data

        for i in np.arange(self.device_run_layers+1):
            if i not in [2, 4, 8]:
                self.layers_gradients_flag_target[i] += 1
        for i in np.arange(self.cloud_run_layers+1):
            if i not in [2, 4, 8]:
                self.layers_gradients_flag_target[i] += 1

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
