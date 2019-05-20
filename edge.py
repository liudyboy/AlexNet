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
        print("process layers: ", process_layers)
        if 1 in process_layers:
            out = self.conv1.forward(out)
        if 2 in process_layers:
            out = self.max_pool1.forward(out)

        ts1 = time.time()
        if 3 in process_layers:
            out = self.conv2.forward(out)
        ts2 = time.time()
        print("lyaer 3 cost time:", (ts2 -ts1) * 1000)

        ts1 = time.time()
        if 4 in process_layers:
            out = self.max_pool2.forward(out)
        ts2 = time.time()
        print("layer 4 cost time: ", (ts2-ts1)*1000.)

        ts1 = time.time()
        if 5 in process_layers:
            out = self.conv3.forward(out)
        ts2 = time.time()
        print("layer 5 cost time: ", (ts2-ts1)*1000.)

        ts1 = time.time()
        if 6 in process_layers:
            out = self.conv4.forward(out)
        ts2 = time.time()
        print("layer 6 cost time: ", (ts2-ts1)*1000.)

        ts1 = time.time()
        if 7 in process_layers:
            out = self.conv5.forward(out)
        ts2 = time.time()
        print("layer 7 cost time: ", (ts2-ts1)*1000.)

        ts1 = time.time()
        if 8 in process_layers:
            out = self.max_pool5.forward(out)
        ts2 = time.time()
        print("layer 8 cost time: ", (ts2-ts1)*1000.)

        ts1 = time.time()
        if 9 in process_layers:
            out = self.fc6.forward(out)
        ts2 = time.time()
        print("layer 9 cost time: ", (ts2-ts1)*1000.)

        ts1 = time.time()
        if 10 in process_layers:
            out = self.fc7.forward(out)
        ts2 = time.time()
        print("layer 10 cost time: ", (ts2-ts1)*1000.)

        ts1 = time.time()
        if 11 in process_layers:
            out = self.fc8.forward(out)
        ts2 = time.time()
        print("layer 11 cost time: ", (ts2-ts1)*1000.)
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

    def update_one_layer_parameters(self, layer_num, batch_size):
        if 1 == layer_num:
            self.conv1.update_parameters(batch=batch_size)
        if 2 == layer_num:
            pass          # it is maxpool layer needn't exchange gradients
        if 3 == layer_num:
            self.conv2.update_parameters(batch=batch_size)
        if 4 == layer_num:
            pass          # it is maxpool layer needn't exchange gradients
        if 5 == layer_num:
            self.conv3.update_parameters(batch=batch_size)
        if 6 == layer_num:
            self.conv4.update_parameters(batch=batch_size)
        if 7 == layer_num:
            self.conv5.update_parameters(batch=batch_size)
        if 8 == layer_num:
            pass          # it is maxpool layer needn't exchange gradients
        if 9 == layer_num:
            self.fc6.update_parameters(batch=batch_size)
        if 10 == layer_num:
            self.fc7.update_parameters(batch=batch_size)
        if 11 == layer_num:
            self.fc8.update_parameters(batch=batch_size)
        return



    # wait the device send the output
    def get_device_output(self):
        while self.device_output_flag is False:
            pass
        return self.device_output_x

    # wait edge return the gradients about X for device
    def get_device_output_gradients(self):
        # print('Try get device output gradients')
        while self.device_output_gradients_flag is False:
            pass
        # print('finish device output gradients')
        return self.device_output_grads

    # Function call by device
    # Parames:  intermediate output, label_Y
    # Return: output gradients
    def process_device_output(self, request, context):
        self.device_output_x = chainer.as_variable(pickle.loads(request.output))
        self.device_output_y = chainer.as_variable(pickle.loads(request.Y))
        self.device_output_flag = True
        grads = self.get_device_output_gradients()

        grads = pickle.dumps(grads.array)
        return communication_pb2.OutputReply(grads=grads)


    def get_cloud_output(self):
        while self.cloud_output_flag is False:
            pass
        return self.cloud_output_x
    def get_cloud_output_gradients(self):
        while self.cloud_output_gradients_flag is False:
            pass
        return self.cloud_output_grads

    # Function call by cloud
    # Parames:  intermediate output, label_Y
    # Return: output gradients
    def process_cloud_output(self, request, context):
        self.cloud_output_x = chainer.as_variable(pickle.loads(request.output))
        self.cloud_output_y = chainer.as_variable(pickle.loads(request.Y))
        self.cloud_output_flag = True
        # print('function process_cloud_output set cloud_output flag: ', self.cloud_output_flag)
        grads = self.get_cloud_output_gradients()

        grads = pickle.dumps(grads.array)
        return communication_pb2.OutputReply(grads=grads)



    def Add_gradients(self, layer_num, grads_w, grads_bias):
        if 1 == layer_num:
            self.conv1.accumulate_params_grad(grads_w, grads_bias)
        if 2 == layer_num:
            pass          # it is maxpool layer needn't exchange gradients
        if 3 == layer_num:
            self.conv2.accumulate_params_grad(grads_w, grads_bias)
        if 4 == layer_num:
            pass          # it is maxpool layer needn't exchange gradients
        if 5 == layer_num:
            self.conv3.accumulate_params_grad(grads_w, grads_bias)
        if 6 == layer_num:
            self.conv4.accumulate_params_grad(grads_w, grads_bias)
        if 7 == layer_num:
            self.conv5.accumulate_params_grad(grads_w, grads_bias)
        if 8 == layer_num:
            pass          # it is maxpool layer needn't exchange gradients
        if 9 == layer_num:
            self.fc6.accumulate_params_grad(grads_w, grads_bias)
        if 10 == layer_num:
            self.fc7.accumulate_params_grad(grads_w, grads_bias)
        if 11 == layer_num:
            self.fc8.accumulate_params_grad(grads_w, grads_bias)
        return


    # get one layer gradients
    def get_gradients(self, layer_num):
        if 1 == layer_num:
            grads_w, grads_bias = self.conv1.get_params_grad()
            return grads_w, grads_bias
        if 2 == layer_num:
            pass          # it is maxpool layer needn't exchange gradients
        if 3 == layer_num:
            grads_w, grads_bias = self.conv2.get_params_grad()
            return grads_w, grads_bias
        if 4 == layer_num:
            pass          # it is maxpool layer needn't exchange gradients
        if 5 == layer_num:
            grads_w, grads_bias = self.conv3.get_params_grad()
            return grads_w, grads_bias
        if 6 == layer_num:
            grads_w, grads_bias = self.conv4.get_params_grad()
            return grads_w, grads_bias
        if 7 == layer_num:
            grads_w, grads_bias = self.conv5.get_params_grad()
            return grads_w, grads_bias
        if 8 == layer_num:
            pass          # it is maxpool layer needn't exchange gradients
        if 9 == layer_num:
            grads_w, grads_bias = self.fc6.get_params_grad()
            return grads_w, grads_bias
        if 10 == layer_num:
            grads_w, grads_bias = self.fc7.get_params_grad()
            return grads_w, grads_bias
        if 11 == layer_num:
            grads_w, grads_bias = self.fc8.get_params_grad()
            return grads_w, grads_bias

    # Function call by cloud or device
    # Parames:  num_layer, grads_w, grads_bias
    # Return: grads_w, grads_bias
    def get_one_layer_gradients(self, request, context):
        num_layer = pickle.loads(request.layer_num)
        grads_w = chainer.as_variable(pickle.loads(request.grads_w))
        grads_bias = chainer.as_variable(pickle.loads(request.grads_bias))
        name = pickle.loads(request.name)

        self.Log('{} call get one layer gradients for layer {}'.format(name, num_layer))
        # ts1 = time.time()
        # wait for edge prepared for receive gradients from others
        while self.prepared_for_recv_gradients == False:
            pass
        # ts2 = time.time()
        # self.Log('{} call for layer {} gradients, wait edge prepared for recv gradients time: {}'.format(name, num_layer, (ts2 - ts1) * 1000.))

        self.Add_gradients(num_layer, grads_w, grads_bias)
        self.layers_gradients_flag[num_layer] += 1


        ts1 = time.time()
        while self.layers_gradients_flag[num_layer] != self.layers_gradients_flag_target[num_layer]:
            pass
        ts2 = time.time()
        self.Log('{} call for layer {} gradients, wait another time: {}'.format(name, num_layer, (ts2 - ts1) * 1000.))
        grads_w, grads_bias = self.get_gradients(num_layer)
        grads_w = pickle.dumps(grads_w.array)
        grads_bias = pickle.dumps(grads_bias.array)

        return communication_pb2.GradsReply(grads_w=grads_w, grads_bias=grads_bias)


    def wait_update_parameters_completed(self):
        while True:
            if (self.layers_gradients_flag == self.layers_gradients_flag_target).all():
                break
            # print('layers gradeints flag :', self.layers_gradients_flag)
            # print('layers gradeints flag target:', self.layers_gradients_flag_target)
        return

    def device_ahead_cloud(self, input_x, label):
        # print('function device_ahead_cloud start')
        # Forward phase
        process_layers = np.arange(self.device_run_layers+1)
        out = self.cal_forward(input_x, process_layers)
        device_send_output = self.get_device_output()
        self.device_batch = device_send_output.shape[0]                   #record the device training batch
        # print('finish function get_device_output')
        out = np.append(out.array, device_send_output.array, axis=0)
        process_layers = np.arange(self.device_run_layers+1, self.cloud_run_layers+1)
        out = self.cal_forward(chainer.as_variable(out), process_layers)
        # print('start function get_cloud_output')
        cloud_send_output = self.get_cloud_output()
        self.cloud_batch = cloud_send_output.shape[0]                # record the cloud  training batch

        # print('finish function get_cloud_output')
        out = np.append(out.array, cloud_send_output.array, axis=0)
        process_layers = np.arange(self.cloud_run_layers+1, 12)
        out = self.cal_forward(chainer.as_variable(out), process_layers)

        # print('start calculate gradeints phase')
        # Calculate gradiens phase
        all_Y = np.append(self.Y.array, self.device_output_y.array, axis=0)
        all_Y = np.append(all_Y, self.cloud_output_y.array, axis=0)

        process_layers = np.arange(self.cloud_run_layers+1, 12)
        out = self.cal_gradients(out, process_layers, chainer.as_variable(all_Y))

        # spilt the cloud gradients
        batch_size = out.shape[0]

        self.cloud_output_grads = out[batch_size-self.cloud_batch:]
        out = out[:batch_size-self.cloud_batch]
        self.cloud_output_gradients_flag = True

        # continue calculate gradients
        process_layers = np.arange(self.device_run_layers+1, self.cloud_run_layers+1)
        out = self.cal_gradients(out, process_layers)

        # print('run before device gradeints flag: ', self.device_output_gradients_flag)
        # spilt the device gradiens
        batch_size = out.shape[0]
        self.device_output_grads = out[batch_size-self.device_batch:]
        out = out[:batch_size-self.device_batch]
        self.device_output_gradients_flag = True

        # print('run after device gradeints flag: ', self.device_output_gradients_flag)
        # continue calculate gradients
        process_layers = np.arange(self.device_run_layers+1)
        out = self.cal_gradients(out, process_layers)

        # update parameters phase
        process_layers = np.arange(self.cloud_run_layers+1, 12)
        self.update_layers_parameters(process_layers, self.TOTAL_BATCH_SZIE)

        self.prepared_for_recv_gradients = True

        self.wait_update_parameters_completed()
        return


    def cloud_ahead_device(self, input_x, label):
        # Forward phase
        process_layers = np.arange(self.cloud_run_layers+1)
        out = self.cal_forward(input_x, process_layers)
        cloud_send_output = self.get_cloud_output()
        self.cloud_batch = cloud_send_output.shape[0]

        out = np.append(out.array, cloud_send_output.array, axis=0)
        process_layers = np.arange(self.cloud_run_layers+1, self.device_run_layers+1)
        out = self.cal_forward(chainer.as_variable(out), process_layers)

        device_send_output = self.get_device_output()
        self.device_batch = device_send_output.shape[0]

        out = np.append(out.array, device_send_output.array, axis=0)
        process_layers = np.arange(self.device_run_layers+1, 12)
        out = self.cal_forward(chainer.as_variable(out), process_layers)

        # Calculate gradients phase
        all_Y = np.append(self.Y.array, self.cloud_output_y.array, axis=0)
        all_Y = np.append(all_Y, self.device_output_y.array, axis=0)

        process_layers = np.arange(self.device_run_layers+1, 12)
        out = self.cal_gradients(out, process_layers, chainer.as_variable(all_Y))

        #spilt the device gradients
        batch_size = out.shape[0]
        self.device_output_grads = out[batch_size-self.device_batch:]
        out = out[:batch_size-self.device_batch]
        self.device_output_gradients_flag = True

        # continue calculate gradients
        process_layers = np.arange(self.cloud_run_layers+1, self.device_run_layers+1)
        out = self.cal_gradients(out, process_layers)

        # spilt the cloud gradients
        batch_size = out.shape[0]
        self.cloud_output_grads = out[batch_size-self.cloud_batch:]
        out = out[:batch_size-self.cloud_batch]
        self.cloud_output_gradients_flag = True

        #continue calculate gradients
        process_layers = np.arange(self.cloud_run_layers+1)
        out = self.cal_gradients(out, process_layers)

        # update parameters phase
        process_layers = np.arange(self.device_run_layers+1, 12)
        self.update_layers_parameters(process_layers, self.TOTAL_BATCH_SZIE)

        self.prepared_for_recv_gradients = True
        self.wait_update_parameters_completed()
        return

    def device_equal_cloud(self, input_x, label):
        # Forward phase
        process_layers = np.arange(self.device_run_layers+1)
        out = self.cal_forward(input_x, process_layers)
        cloud_send_output = self.get_cloud_output()
        device_send_output = self.get_device_output()


        self.cloud_batch = cloud_send_output.shape[0]
        self.device_batch = device_send_output.shape[0]

        out = np.append(out.array, cloud_send_output.array, axis=0)
        out = np.append(out, device_send_output.array, axis=0)


        # ts1 = time.time()
        process_layers = np.arange(self.device_run_layers+1, 12)
        out = self.cal_forward(chainer.as_variable(out), process_layers)


        # Calculate gradients phase
        all_Y = np.append(self.Y.array, self.cloud_output_y.array, axis=0)
        all_Y = np.append(all_Y, self.device_output_y.array, axis=0)

        process_layers = np.arange(self.device_run_layers+1, 12)
        out = self.cal_gradients(out, process_layers, chainer.as_variable(all_Y))

        # Spilt gradients for cloud and device
        batch_size = out.shape[0]
        self.device_output_grads = out[batch_size-self.device_batch:]
        out = out[:batch_size-self.device_batch]
        self.device_output_gradients_flag = True

        batch_size = out.shape[0]
        self.cloud_output_grads = out[batch_size-self.cloud_batch:]
        out = out[:batch_size-self.cloud_batch]
        self.cloud_output_gradients_flag = True

        # Continue calculate gradients
        process_layers = np.arange(self.device_run_layers+1)
        out = self.cal_gradients(out, process_layers)

        # Update parameters phase
        process_layers = np.arange(self.device_run_layers+1, 12)
        self.update_layers_parameters(process_layers, self.TOTAL_BATCH_SZIE)

        self.prepared_for_recv_gradients = True
        self.wait_update_parameters_completed()

        return




        return
    def run_training(self, input_x, label):
        # condition 1: device run layers less than cloud run layers
        if self.device_run_layers < self.cloud_run_layers:
            self.device_ahead_cloud(input_x, label)

        elif self.device_run_layers > self.cloud_run_layers:
            self.cloud_ahead_device(input_x, label)

        elif self.device_run_layers == self.cloud_run_layers:
            self.device_equal_cloud(input_x, label)

    def get_singal_for_new_epoch(self, request, context):
        singal = pickle.dumps(self.ready_for_new_epoch)
        return communication_pb2.Singal(singal=singal)

    def get_singal_for_finished_epoch(self, request, context):
        singal = pickle.dumps(self.finished_epoch)
        return communication_pb2.Singal(singal=singal)


    # Function call by device
    # Parames: raw_data_x, layer_y
    def process_raw_data(self, request, context):
        if self.init_layers_flag is False:
            print('Initial Model layers')
            self.device_run_layers, self.cloud_run_layers = args.args_prase()
            edge_run_layers = 11
            self.process_layers = np.arange(edge_run_layers+1)
            self.init_layers(self.process_layers)

        # print('process a new raw data')
        self.init_variables()

        raw_input = chainer.as_variable(pickle.loads(request.raw_x))
        self.Y = chainer.as_variable(pickle.loads(request.Y))
        out = self.run_training(raw_input, self.Y)

        self.ready_for_new_epoch = False

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
