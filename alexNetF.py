
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



conv_stride = [4, 4]
conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride)
conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1])
conv1 = layers.conv2d(filters=96, kernel=[11, 11], padding='SAME', name='conv1', activation='relu', normalization='local_response_normalization', stride=conv_stride)
conv2 = layers.conv2d(filters=256, kernel=[5, 5], padding='SAME', name='conv2', activation='relu', normalization="local_response_normalization", stride=[1, 1])
conv3 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv3', activation='relu', stride=[1, 1])
conv4 = layers.conv2d(filters=384, kernel=[3, 3], padding='SAME', name='conv4', activation='relu', stride=[1, 1])
conv5 = layers.conv2d(filters=256, kernel=[3, 3], padding='SAME', name='conv5', activation='relu', stride=[1, 1])

fc6 = layers.dense(4096, activation='relu', dropout=True, name='fc6')
fc7 = layers.dense(4096, activation='relu', dropout=True, name='fc7')
fc8 = layers.dense(1000, activation='relu', name='fc8')

max_pool1 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
max_pool2 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])
max_pool5 = layers.max_pool2d(ksize=[3, 3], stride=[2, 2])

def Forward(x):
    out = conv1.forward(x)
    # print('layer 1:', out.shape)
    out = max_pool1.forward(out)
    
    # print('layer max 1:', out.shape)
    
    out = conv2.forward(out)
    # print('layer conv2:', out.shape)
    out = max_pool2.forward(out)
    
    # print('layer max pool 2:', out.shape)
    

    out = conv3.forward(out)
    
    
    # print('layer 3:', out.shape)
    out = conv4.forward(out)
    
    # print('layer 4:', out.shape)
    out = conv5.forward(out)

    # print('layer 5:', out.shape)
    out = max_pool5.forward(out)
    # print('layer max 5', out.shape)

    out = fc6.forward(out)
    # print('layer fc6', out.shape)

    out = fc7.forward(out)

    # print('layer fc7:', out.shape)
    out = fc8.forward(out)


    return out

def Backward(loss, temp_out):
    ts = time.time()
    end_time = time.ctime(ts)
    print("backward start time:", end_time)

    d_out = chainer.grad([loss], [temp_out])
    
    if isinstance(d_out, (list)):
        d_out = d_out[0]
    d_out = fc8.backward(d_out)
        
    ts = time.time()
    end_time = time.ctime(ts)
    print("fc8 end time:", end_time)
        
    d_out = fc7.backward(d_out)

    ts = time.time()
    end_time = time.ctime(ts)
    print("fc7 end time:", end_time)

    d_out = fc6.backward(d_out)
    
    ts = time.time()
    end_time = time.ctime(ts)
    print("fc6 end time:", end_time)

    d_out = max_pool5.backward(d_out)
    ts = time.time()
    end_time = time.ctime(ts)
    print("max_pool5 end time:", end_time)


    d_out = conv5.backward(d_out)

    ts = time.time()
    end_time = time.ctime(ts)
    print("conv5 end time:", end_time)

    d_out = conv4.backward(d_out)
    ts = time.time()
    end_time = time.ctime(ts)
    print("conv4 end time:", end_time)


    d_out = conv3.backward(d_out)
    ts = time.time()
    end_time = time.ctime(ts)
    print("conv3 end time:", end_time)

    d_out = max_pool2.backward(d_out)
    ts = time.time()
    end_time = time.ctime(ts)
    print("max_pool2 end time:", end_time)
    
    
    
    d_out = conv2.backward(d_out)
    ts = time.time()
    end_time = time.ctime(ts)
    print("conv2 end time:", end_time)

    d_out = max_pool1.backward(d_out)
    ts = time.time()
    end_time = time.ctime(ts)
    print("max_Pool1 end time:", end_time)



    d_out = conv1.backward(d_out)
    ts = time.time()
    end_time = time.ctime(ts)
    print("conv1 end time:", end_time)

    del d_out


generations = 1000

batch_size = 128


ts = time.time()
start_time = time.ctime(ts)
print("start time:", start_time)
gc.set_threshold(1, 1, 1)

file = open('temp.txt', 'w')
file.close()

for i in range(generations):
    trainX, trainY = utils.get_batch_data(batch_size)
    
    Y = trainY.astype(np.int32)
    
    print('trainX shape:', trainX.shape)
    temp_out = Forward(trainX)

    ts = time.time()
    infer_end_time = time.ctime(ts)
    print("infer_end time:", infer_end_time)
    # file = open('temp.txt', 'a+')
    # file.write('epoch {}:'.format(i))
    # file.write('temp_out: {}'.format(temp_out.array))
    # file.close()

    loss = F.softmax_cross_entropy(temp_out, Y)
    accuracy = F.accuracy(temp_out, Y)
    print('epoch #{} loss: {}'.format(i, loss))
    print('epoch #{} accuracy: {}'.format(i, accuracy))


    Backward(loss, temp_out)
    ts = time.time()
    end_time = time.ctime(ts)
    print("end time:", end_time)

    del trainX, trainY, Y
    gc.collect()

    file = open('temp.txt', 'a+')
    file.write('epoch {}:'.format(i))
    file.write(str(gc.garbage))
    file.close()

