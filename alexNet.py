import numpy as np
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
import layers
import utils
import gc
import time

gpu_flag = True

class AlexNet():
    insize = 227

    def __init__(self):

        conv_stride = [4, 4]
        self.conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride, use_gpu=gpu_flag)
        self.conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1], use_gpu=gpu_flag)
        self.conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1], use_gpu=gpu_flag)
        self.conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1], use_gpu=gpu_flag)
        self.conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1], use_gpu=gpu_flag)

        self.fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6', use_gpu=gpu_flag)
        self.fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7', use_gpu=gpu_flag)
        self.fc8 = layers.dense(1000, activation='relu', name='fc8', use_gpu=gpu_flag)

        self.max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        self.max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
        self.max_pool5 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])


        self.init_file()

        self.layer1_time = 0.
        self.layer2_time = 0.
        self.layer3_time = 0.
        self.layer4_time = 0.
        self.layer5_time = 0.
        self.layer6_time = 0.
        self.layer7_time = 0.
        self.layer8_time = 0.
        self.layer9_time = 0.
        self.layer10_time = 0.
        self.layer11_time = 0.


    def forward(self, x):
        now1 = time.time()

        out = self.conv1.forward(x)
        
        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer1_time = cost
        

        now1 = time.time()

        out = self.max_pool1.forward(out)
        
        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer2_time = cost

        now1 = time.time()

        out = self.conv2.forward(out)
        
        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer3_time = cost



        now1 = time.time()

        out = self.max_pool2.forward(out)
        
        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer4_time = cost


        now1 = time.time()

        out = self.conv3.forward(out)
        
        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer5_time = cost


        now1 = time.time()

        out = self.conv4.forward(out)
        
        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer6_time = cost

        now1 = time.time()

        out = self.conv5.forward(out)
        
        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer7_time = cost


        now1 = time.time()

        out = self.max_pool5.forward(out)
        
        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer8_time = cost
        now1 = time.time()

        # out.to_cpu()
        out = self.fc6.forward(out)
        
        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer9_time = cost
        now1 = time.time()

        out = self.fc7.forward(out)

        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer10_time = cost
        now1 = time.time()

        out = self.fc8.forward(out)

        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer11_time = cost

        if gpu_flag is True:
            out.to_cpu()
        return out



    def backward(self, loss, temp_out):
        file = open("layer11.txt", "a+")
        now1 = time.time()

        d_out = chainer.grad([loss], [temp_out])

        if isinstance(d_out, (list)):
            d_out = d_out[0]

        d_out = self.fc8.backward(d_out)

        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer11_time += cost
        file.write(str(self.layer11_time))
        file.write('\n')
        file.close()

        file = open("layer10.txt", "a+")
        now1 = time.time()

        d_out = self.fc7.backward(d_out)

        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer10_time += cost

        file.write(str(self.layer10_time))
        file.write('\n')
        file.close()


        file = open("layer9.txt", "a+")
        now1 = time.time()

        d_out = self.fc6.backward(d_out)

        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer9_time += cost
        file.write(str(self.layer9_time))
        file.write('\n')
        file.close()

        file = open("layer8.txt", "a+")
        now1 = time.time()
        # d_out.to_gpu(0)

        d_out = self.max_pool5.backward(d_out)

        now2 = time.time()
        cost = (now2 - now1) * 1000.0

        self.layer8_time += cost
        file.write(str(self.layer8_time))
        file.write('\n')
        file.close()


        file = open("layer7.txt", "a+")
        now1 = time.time()

        d_out = self.conv5.backward(d_out)

        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer7_time += cost
        file.write(str(self.layer7_time))
        file.write('\n')
        file.close()

        file = open("layer6.txt", "a+")
        now1 = time.time()

        d_out = self.conv4.backward(d_out)

        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer6_time += cost
        file.write(str(self.layer6_time))
        file.write('\n')
        file.close()


        file = open("layer5.txt", "a+")
        now1 = time.time()

        d_out = self.conv3.backward(d_out)

        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer5_time += cost
        file.write(str(self.layer5_time))
        file.write('\n')
        file.close()

        file = open("layer4.txt", "a+")
        now1 = time.time()

        d_out = self.max_pool2.backward(d_out)

        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer4_time += cost
        file.write(str(self.layer4_time))
        file.write('\n')
        file.close()

        file = open("layer3.txt", "a+")
        now1 = time.time()

        d_out = self.conv2.backward(d_out)

        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer3_time += cost
        file.write(str(self.layer3_time))
        file.write('\n')
        file.close()

        file = open("layer2.txt", "a+")
        now1 = time.time()

        d_out = self.max_pool1.backward(d_out)

        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer2_time += cost

        file.write(str(self.layer2_time))
        file.write('\n')
        file.close()

        file = open("layer1.txt", "a+")
        now1 = time.time()

        d_out = self.conv1.backward(d_out)

        now2 = time.time()
        cost = (now2 - now1) * 1000.0
        self.layer1_time += cost
        file.write(str(self.layer1_time))
        file.write('\n')
        file.close()


    def init_file(self):
        file = open("layer1.txt", "w")
        file.close()

        file = open("layer2.txt", "w")
        file.close()

        file = open("layer3.txt", "w")
        file.close()

        file = open("layer4.txt", "w")
        file.close()

        file = open("layer5.txt", "w")
        file.close()

        file = open("layer6.txt", "w")
        file.close()

        file = open("layer7.txt", "w")
        file.close()

        file = open("layer8.txt", "w")
        file.close()

        file = open("layer9.txt", "w")
        file.close()

        file = open("layer10.txt", "w")
        file.close()

        file = open("layer11.txt", "w")
        file.close()



generations = 100

batch_size = 128

alexNet = AlexNet()

ts = time.time()
start_time = time.ctime(ts)
print("start time:", start_time)
for i in range(generations):
    trainX, trainY = utils.get_batch_data(batch_size)
    
    Y = trainY.astype(np.int32)
    
    temp_out = alexNet.forward(trainX)

    ts = time.time()
    infer_end_time = time.ctime(ts)
    print("infer_end time:", infer_end_time)

    loss = F.softmax_cross_entropy(temp_out, Y)
    accuracy = F.accuracy(temp_out, Y)
    print('epoch #{} loss: {}'.format(i, loss))
    print('epoch #{} accuracy: {}'.format(i, accuracy))


    alexNet.backward(loss, temp_out)
    ts = time.time()
    end_time = time.ctime(ts)
    print("end time:", end_time)

    del trainX, trainY, Y

