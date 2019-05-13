from __future__ import print_function
import logging
import numpy as np

import grpc
import communication_pb2
import communication_pb2_grpc
import pickle
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
import sys
import args


#  send input raw data to destination
# destination: "ip:port"
def conn_send_input(destination, sendArray, Y):
    with grpc.insecure_channel(destination, options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)]) as channel:
        stub = communication_pb2_grpc.CommStub(channel)
        sendArray = pickle.dumps(sendArray)
        Y = pickle.dumps(Y)
        recv_array = stub.process_raw_data(communication_pb2.ArrayRecv(array=sendArray, Y=Y))
        recv_array = pickle.loads(recv_array.array)
    return recv_array

# send output to destination
def conn_device_send_output(destination, sendArray, Y):
    with grpc.insecure_channel(destination, options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)]) as channel:
        stub = communication_pb2_grpc.CommStub(channel)
        sendArray = pickle.dumps(sendArray)
        Y = pickle.dumps(Y)
        recv_array = stub.process_device_output(communication_pb2.ArrayRecv(array=sendArray, Y=Y))
        recv_array = pickle.loads(recv_array.array)
    return recv_array
# send gradient to destination
def conn_send_gradient(destination, sendArray, Y):
    with grpc.insecure_channel(destination, options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)]) as channel:
        stub = communication_pb2_grpc.CommStub(channel)
        sendArray = pickle.dumps(sendArray)
        Y = pickle.dumps(Y)
        recv_array = stub.Forwarding(communication_pb2.ArrayRecv(array=sendArray, Y=Y))
        recv_array = pickle.loads(recv_array.array)
    return recv_array

