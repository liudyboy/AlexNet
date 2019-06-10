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
from model import LeNet5


_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class Connecter(communication_pb2_grpc.CommServicer):
    init_layers_flag = False
    finished_epoch = False
    ready_for_new_epoch = False
    device_output_flag = False
    device_output_gradients_flag = False
    cloud_output_flag = False
    cloud_output_gradients_flag = False
    model_layers_num = 7
    def init_variables(self):
        self.TOTAL_BATCH_SZIE = 512
        self.device_output_grads = None
        self.cloud_output_grads = None
        self.layers_gradients_flag = np.zeros(self.model_layers_num+1)
        self.layers_gradients_flag[0] = 2            # for target[0] initial value is 1
        self.layers_gradients_flag_target = np.zeros(self.model_layers_num+1)
        self.prepared_for_recv_gradients = False


        for i in np.arange(self.device_run_layers+1):
            if i not in [2, 4]:
                self.layers_gradients_flag_target[i] += 1
        for i in np.arange(self.cloud_run_layers+1):
            if i not in [2, 4]:
                self.layers_gradients_flag_target[i] += 1

    def init_flag(self):
        self.cloud_output_gradients_flag = False
        self.device_output_gradients_flag = False
        self.cloud_output_flag = False
        self.device_output_flag = False


    # wait the device send the output
    def get_device_output(self):
        while self.device_output_flag is False:
            pass
        return self.device_output_x

    # wait edge return the gradients about X for device
    def get_device_output_gradients(self):
        print('Try get device output gradients')
        while self.device_output_gradients_flag is False:
            pass
        print('finish device output gradients')
        return self.device_output_grads

    # Function call by device
    # Parames:  intermediate output, label_Y
    # Return: output gradients
    def process_device_output(self, request, context):
        self.device_output_x = chainer.as_variable(pickle.loads(request.output))
        self.device_output_y = chainer.as_variable(pickle.loads(request.Y))
        self.device_output_flag = True
        ts1 = time.time()
        print("get device output time: ", ts1)
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
        ts1 = time.time()
        print("Get cloud output time: ", ts1)

        grads = self.get_cloud_output_gradients()

        grads = pickle.dumps(grads.array)
        return communication_pb2.OutputReply(grads=grads)


    # Function call by cloud or device
    # Parames:  num_layer, grads_w, grads_bias
    # Return: grads_w, grads_bias
    def get_one_layer_gradients(self, request, context):
        num_layer = pickle.loads(request.layer_num)
        grads_w = chainer.as_variable(pickle.loads(request.grads_w))
        grads_bias = chainer.as_variable(pickle.loads(request.grads_bias))
        name = pickle.loads(request.name)

        # self.Log('{} call get one layer gradients for layer {}'.format(name, num_layer))
        while self.prepared_for_recv_gradients == False:
            pass
        # self.Log('prepared for recv gradients is: {}'.format(self.prepared_for_recv_gradients))
        self.mnist.add_params_grads(num_layer, grads_w, grads_bias)
        self.layers_gradients_flag[num_layer] += 1

        # self.Log("gradients flag compare to gradients target")
        ts1 = time.time()
        while self.layers_gradients_flag[num_layer] != self.layers_gradients_flag_target[num_layer]:
            pass
        ts2 = time.time()
        # self.Log('{} call for layer {} gradients, wait another time: {}'.format(name, num_layer, (ts2 - ts1) * 1000.))
        grads_w, grads_bias = self.mnist.get_params_grads(num_layer)

        grads_w = pickle.dumps(grads_w.array)
        grads_bias = pickle.dumps(grads_bias.array)

        return communication_pb2.GradsReply(grads_w=grads_w, grads_bias=grads_bias)


    def wait_update_parameters_completed(self):
        while True:
            if (self.layers_gradients_flag == self.layers_gradients_flag_target).all():
                break
        return

    def device_ahead_cloud(self, input_x, label):
        # Forward phase
        ts1 = time.time()
        process_layers = np.arange(1, self.device_run_layers+1)
        out = self.mnist.forward(input_x, process_layers)
        ts2 = time.time()
        t1 = (ts2 - ts1) * 1000
        if self.device_run_layers != 0:
            device_send_output = self.get_device_output()
            self.device_batch = device_send_output.shape[0]                   #record the device training batch
            out = np.append(out.array, device_send_output.array, axis=0)

        ts1 = time.time()
        process_layers = np.arange(self.device_run_layers+1, self.cloud_run_layers+1)
        out = self.mnist.forward(chainer.as_variable(out), process_layers)
        ts2 = time.time()
        t1 = t1 + (ts2 - ts1) * 1000
        cloud_send_output = self.get_cloud_output()
        self.cloud_batch = cloud_send_output.shape[0]                # record the cloud  training batch
        ts1 = time.time()
        out = np.append(out.array, cloud_send_output.array, axis=0)
        process_layers = np.arange(self.cloud_run_layers+1, self.model_layers_num+1)
        out = self.mnist.forward(chainer.as_variable(out), process_layers)
        ts2 = time.time()
        t1 = t1 + (ts2 - ts1) * 1000
        self.Log("edge forward cost time {}".format(t1))
        # Calculate gradiens phase
        ts1 = time.time()
        if self.device_run_layers != 0:
            all_Y = np.append(self.Y.array, self.device_output_y.array, axis=0)
            all_Y = np.append(all_Y, self.cloud_output_y.array, axis=0)
        else:
            all_Y = np.append(self.Y.array, self.cloud_output_y.array, axis=0)

        process_layers = np.arange(self.cloud_run_layers+1, self.model_layers_num+1)
        out = self.mnist.cal_gradients(out, process_layers, chainer.as_variable(all_Y))
        ts2 = time.time()
        t2 = (ts2 - ts1) * 1000
        # spilt the cloud gradients
        batch_size = out.shape[0]

        self.cloud_output_grads = out[batch_size-self.cloud_batch:]
        out = out[:batch_size-self.cloud_batch]
        self.cloud_output_gradients_flag = True

        ts1 = time.time()
        # continue calculate gradients
        process_layers = np.arange(self.device_run_layers+1, self.cloud_run_layers+1)
        out = self.mnist.cal_gradients(out, process_layers)
        ts2 = time.time()
        t2 = t2 + (ts2 - ts1) * 1000
        # spilt the device gradiens
        if self.device_run_layers != 0:
            batch_size = out.shape[0]
            self.device_output_grads = out[batch_size-self.device_batch:]
            out = out[:batch_size-self.device_batch]
            self.device_output_gradients_flag = True

            ts1 = time.time()
            # continue calculate gradients
            process_layers = np.arange(1, self.device_run_layers+1)
            out = self.mnist.cal_gradients(out, process_layers)
            ts2 = time.time()
            t2 = t2 + (ts2 - ts1) * 1000
            self.Log("edge cal gradient time {}".format(t2))

        # update parameters phase
        process_layers = np.arange(self.cloud_run_layers+1, self.model_layers_num+1)
        self.mnist.update_layers_parameters(process_layers, self.TOTAL_BATCH_SZIE)

        self.prepared_for_recv_gradients = True

        self.wait_update_parameters_completed()
        return


    def cloud_ahead_device(self, input_x, label):
        # Forward phase
        process_layers = np.arange(1, self.cloud_run_layers+1)
        out = self.mnist.forward(input_x, process_layers)

        if self.cloud_run_layers != 0:
            cloud_send_output = self.get_cloud_output()
            self.cloud_batch = cloud_send_output.shape[0]
            out = np.append(out.array, cloud_send_output.array, axis=0)

        process_layers = np.arange(self.cloud_run_layers+1, self.device_run_layers+1)
        out = self.mnist.forward(chainer.as_variable(out), process_layers)

        device_send_output = self.get_device_output()
        self.device_batch = device_send_output.shape[0]

        out = np.append(out.array, device_send_output.array, axis=0)
        process_layers = np.arange(self.device_run_layers+1, self.model_layers_num+1)
        out = self.mnist.forward(chainer.as_variable(out), process_layers)

        # Calculate gradients phase
        if self.cloud_run_layers != 0:
            all_Y = np.append(self.Y.array, self.cloud_output_y.array, axis=0)
            all_Y = np.append(all_Y, self.device_output_y.array, axis=0)
        else:
            all_Y = np.append(self.Y.array, self.device_output_y.array, axis=0)


        process_layers = np.arange(self.device_run_layers+1, self.model_layers_num+1)
        out = self.mnist.cal_gradients(out, process_layers, chainer.as_variable(all_Y))

        #spilt the device gradients
        batch_size = out.shape[0]
        self.device_output_grads = out[batch_size-self.device_batch:]
        out = out[:batch_size-self.device_batch]
        self.device_output_gradients_flag = True

        # continue calculate gradients
        process_layers = np.arange(self.cloud_run_layers+1, self.device_run_layers+1)
        out = self.mnist.cal_gradients(out, process_layers)

        # spilt the cloud gradients
        if self.cloud_run_layers != 0:
            batch_size = out.shape[0]
            self.cloud_output_grads = out[batch_size-self.cloud_batch:]
            out = out[:batch_size-self.cloud_batch]
            self.cloud_output_gradients_flag = True

            #continue calculate gradients
            process_layers = np.arange(1, self.cloud_run_layers+1)
            out = self.mnist.cal_gradients(out, process_layers)

        # update parameters phase
        process_layers = np.arange(self.device_run_layers+1, self.model_layers_num+1)
        self.mnist.update_layers_parameters(process_layers, self.TOTAL_BATCH_SZIE)

        self.prepared_for_recv_gradients = True
        self.wait_update_parameters_completed()
        return

    def device_equal_cloud(self, out, label):
        # Forward phase
        if self.device_run_layers != 0 and self.cloud_run_layers != 0:
            process_layers = np.arange(1, self.device_run_layers+1)
            out = self.mnist.forward(out, process_layers)
            cloud_send_output = self.get_cloud_output()
            device_send_output = self.get_device_output()

            self.cloud_batch = cloud_send_output.shape[0]
            self.device_batch = device_send_output.shape[0]

            out = np.append(out.array, cloud_send_output.array, axis=0)
            out = np.append(out, device_send_output.array, axis=0)

        # ts1 = time.time()
        process_layers = np.arange(self.device_run_layers+1, self.model_layers_num+1)
        out =  self.mnist.forward(chainer.as_variable(out), process_layers)


        # Calculate gradients phase

        if self.device_run_layers != 0 and self.cloud_run_layers != 0:
            all_Y = np.append(self.Y.array, self.cloud_output_y.array, axis=0)
            all_Y = np.append(all_Y, self.device_output_y.array, axis=0)
        else:
            all_Y = self.Y.array
        process_layers = np.arange(self.device_run_layers+1, self.model_layers_num+1)
        out = self.mnist.cal_gradients(out, process_layers, chainer.as_variable(all_Y))

        # Spilt gradients for cloud and device
        if self.device_run_layers != 0 and self.cloud_run_layers != 0:
            batch_size = out.shape[0]
            self.device_output_grads = out[batch_size-self.device_batch:]
            out = out[:batch_size-self.device_batch]
            self.device_output_gradients_flag = True

            batch_size = out.shape[0]
            self.cloud_output_grads = out[batch_size-self.cloud_batch:]
            out = out[:batch_size-self.cloud_batch]
            self.cloud_output_gradients_flag = True

            # Continue calculate gradients
            process_layers = np.arange(1, self.device_run_layers+1)
            out = self.mnist.cal_gradients(out, process_layers)

        # Update parameters phase
        process_layers = np.arange(self.device_run_layers+1, self.model_layers_num+1)
        self.mnist.update_layers_parameters(process_layers, self.TOTAL_BATCH_SZIE)

        self.prepared_for_recv_gradients = True
        self.wait_update_parameters_completed()

        return

    def run_training(self, input_x, label):
        # condition 1: device run layers less than cloud run layers
        if self.device_run_layers < self.cloud_run_layers:
            self.device_ahead_cloud(input_x, label)

        elif self.device_run_layers > self.cloud_run_layers:
            self.cloud_ahead_device(input_x, label)

        elif self.device_run_layers == self.cloud_run_layers:
            self.device_equal_cloud(input_x, label)
        return



    # Function call by device
    # Parames: raw_data_x, layer_y
    def process_raw_data(self, request, context):
        if self.init_layers_flag is False:
            print('Initial Model layers')
            my_args = args.args_prase()
            self.device_run_layers, self.cloud_run_layers = my_args.M1, my_args.M2
            edge_run_layers = self.model_layers_num
            self.process_layers = np.arange(1, edge_run_layers+1)
            self.mnist = LeNet5()
            self.mnist.init_layers(self.process_layers)
            self.init_layers_flag = True

        # print('process a new raw data')
        self.init_variables()
        ts1 = time.time()
        print("Get raw time: ", ts1)
        raw_input = chainer.as_variable(pickle.loads(request.raw_x))
        self.Y = chainer.as_variable(pickle.loads(request.Y))
        out = self.run_training(raw_input, self.Y)

        self.init_flag()
        finsh_signal = pickle.dumps(np.zeros(1))
        return communication_pb2.RawReply(signal=finsh_signal)

    def Log(self, message):
        print(message)
        # pass
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
