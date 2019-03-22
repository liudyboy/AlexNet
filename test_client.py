import rpyc
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
import sys
import pickle


conn = rpyc.connect('192.168.1.153', 18871, config = rpyc.core.protocol.DEFAULT_CONFIG)
input = np.ones(shape=(10000, 1000), dtype=np.float32)
input = pickle.dumps(input)
size = sys.getsizeof(input)
print("send data size:", size/1024./1024.)

tc1 = time.time()
output = conn.root.forward(input)
tc2 = time.time()

print("transfer time:", (tc2 - tc1) * 1000.)
