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


#  Send: input raw data to destination
# destination: "ip:port"
# Return: singal
def conn_send_raw_data(destination, raw_x, Y):
    with grpc.insecure_channel(destination, options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)]) as channel:
        stub = communication_pb2_grpc.CommStub(channel)
        raw_x = pickle.dumps(raw_x)
        Y = pickle.dumps(Y)
        recv_data = stub.process_raw_data(communication_pb2.RawSend(raw_x=raw_x, Y=Y))
        singal = pickle.loads(recv_data.signal)
    return singal

# Send: layer output, Y
#Reture: gradients_x
def conn_send_device_output_data(destination, output, Y):
    with grpc.insecure_channel(destination, options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)]) as channel:
        stub = communication_pb2_grpc.CommStub(channel)
        output = pickle.dumps(output)
        Y = pickle.dumps(Y)
        recv_data = stub.process_device_output(communication_pb2.OutputSend(output=output, Y=Y))
        grads_x = pickle.loads(recv_data.grads)
    return grads_x


# Send: one layer grad_w and grads_bias, layer number
# Return: one laeyr grad_w and grads_bias
def conn_get_gradients(destination, grads_w, grads_bias, layer_num, name):
    with grpc.insecure_channel(destination, options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)]) as channel:
        stub = communication_pb2_grpc.CommStub(channel)
        grads_w = pickle.dumps(grads_w)
        grads_bias = pickle.dumps(grads_bias)
        layer_num = pickle.dumps(layer_num)
        name = pickle.dumps(name)
        recv_data = stub.get_one_layer_gradients(communication_pb2.GradsSend(grads_w=grads_w, grads_bias=grads_bias, layer_num=layer_num, name=name))
        grads_w = pickle.loads(recv_data.grads_w)
        grads_bias = pickle.loads(recv_data.grads_bias)
    return grads_w, grads_bias


# Send: layer output, Y
#Reture: gradients_x
def conn_send_cloud_output_data(destination, output, Y):
    with grpc.insecure_channel(destination, options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)]) as channel:
        stub = communication_pb2_grpc.CommStub(channel)
        output = pickle.dumps(output)
        Y = pickle.dumps(Y)
        recv_data = stub.process_cloud_output(communication_pb2.OutputSend(output=output, Y=Y))
        grads_x = pickle.loads(recv_data.grads)
    return grads_x

def conn_get_singal_for_new_epoch(destination):
    with grpc.insecure_channel(destination, options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)]) as channel:
        stub = communication_pb2_grpc.CommStub(channel)
        recv_data = stub.get_singal_for_new_epoch(communication_pb2.Singal(singal=pickle.dumps(1)))
        singal = pickle.loads(recv_data.singal)
    return singal

def conn_get_singal_for_finished_epoch(destination):
    with grpc.insecure_channel(destination, options=[('grpc.max_message_length', 1024*1024*1024), ('grpc.max_send_message_length', 1024*1024*1024), ('grpc.max_receive_message_length', 1024*1024*1024)]) as channel:
        stub = communication_pb2_grpc.CommStub(channel)
        recv_data = stub.get_singal_for_finished_epoch(communication_pb2.Singal(singal=pickle.dumps(1)))
        singal = pickle.loads(recv_data.singal)
    return singal
