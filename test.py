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


# gc.set_debug(gc.DEBUG_SAVEALL)
# file = open("debug.txt", "w")
# file.close()

fc = layers.dense(1080)
for i in np.arange(100000):
    print('epoch ', i)
    x = np.ones(shape=(128, 10000), dtype=np.float32)
    # result = fc.forward(x)
    # print(result.shape)
    dout = chainer.as_variable(np.ones(shape=(128, 1080), dtype=np.float32))
    fc.backward(dout)

    # file = open("debug.txt", "a")
    # file.write(str(gc.garbage))
    # file.close()


# x = chainer.as_variable(np.ones(shape=(100, 100), dtype=np.float32))
# y = chainer.as_variable(0.1 * np.ones(shape=(100, 100), dtype=np.float32))
# z = update.vanilla_update(x, y)

# print(x)
# print(z)
