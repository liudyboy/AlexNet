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


rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
conn = rpyc.connect('192.168.1.153', 18871, config = rpyc.core.protocol.DEFAULT_CONFIG)
conn.root.forward()
