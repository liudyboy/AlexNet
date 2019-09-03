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
import cifar_utils as utils
import gc
import time
import numpy as np
import pickle
import sys
import args
from model import LeNet5
import connect


_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class Connecter(communication_pb2_grpc.CommServicer):
    init_layers_flag = False
    cloud_address = '192.168.1.70:50055'

    #parames: a array of numpy 
    # return: a quantize array
    def quantization(self, array):
        Max = np.amax(array)
        Min = np.amin(array)
        denominator = Max - Min
        numerator = (pow(2., 8.)-1.) * (array - Min)
        quan_array = numerator/denominator
        quan_array = np.array(quan_array, dtype=np.uint8)
        return quan_array
    def send_output_data(self, destination, output, Y):
        ts1 = time.time()
        print('Send output to Cloud start time: ', ts1)
        array = self.quantization(output.array)
        reply = connect.conn_send_cloud_output_data(destination, array, Y.array)
        reply = np.array(reply, dtype=np.float32)
        reply = chainer.as_variable(reply)
        ts2 = time.time()
        print("edge send output to cloud cost time: ", (ts2 - ts1) * 1000.)
        return reply


    # Function call by device
    # Parames: raw_data_x, layer_y
    def process_raw_data(self, request, context):
        if self.init_layers_flag is False:
            print('initial the Model Layers')
            edge_run_layers = 2
            self.lenet = LeNet5()
            self.process_layers = np.arange(1, edge_run_layers+1)
            self.lenet.init_layers(self.process_layers)

        raw_input = chainer.as_variable(pickle.loads(request.raw_x))
        self.Y = chainer.as_variable(pickle.loads(request.Y))
        batch_size = raw_input.shape[0]
        out = self.lenet.forward(raw_input, self.process_layers)
        output_reply = self.send_output_data(self.cloud_address, out, self.Y)
        out = self.lenet.cal_gradients(output_reply, self.process_layers, self.Y)
        self.lenet.update_layers_parameters(self.process_layers, batch_size)

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
