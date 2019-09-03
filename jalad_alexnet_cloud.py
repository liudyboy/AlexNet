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
from model import AlexNet
import connect
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
class Connecter(communication_pb2_grpc.CommServicer):
    init_layers_flag = False
    model_layers_num = 11

    def quantization(self, array):
        Max = np.amax(array)
        Min = np.amin(array)
        denominator = Max - Min
        numerator = (pow(2., 8.)-1.) * (array - Min)
        quan_array = numerator/denominator
        quan_array = np.array(quan_array, dtype=np.uint8)
        return quan_array
    # Function call by cloud
    # Parames:  intermediate output, label_Y
    # Return: output gradients
    def process_cloud_output(self, request, context):
        if self.init_layers_flag is False:
            print('initial layers')
            paritition_point = 2
            self.use_gpu = True
            self.alexnet = AlexNet(self.use_gpu)
            self.process_layers = np.arange(paritition_point+1, self.model_layers_num+1)
            self.init_layers_flag = True
            self.alexnet.init_layers(self.process_layers)
        array = np.array(pickle.loads(request.output), dtype=np.float32)
        self.edge_output_x = chainer.as_variable(array)
        self.edge_output_y = chainer.as_variable(pickle.loads(request.Y))

        ts1 = time.time()
        print("Get Edge output time: ", ts1)
        output = self.alexnet.forward(self.edge_output_x, self.process_layers)
        output_reply = self.alexnet.cal_gradients(output, self.process_layers, self.edge_output_y)
        self.alexnet.update_layers_parameters(self.process_layers)
        grads = output_reply
        grads.to_cpu()
        grads = self.quantization(grads.array)
        grads = pickle.dumps(grads)
        return communication_pb2.OutputReply(grads=grads)

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
