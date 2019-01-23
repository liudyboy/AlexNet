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
import utils



def vanilla_update(params, grads, learning_rate=0.1):

    # print('previous {}'.format(params))

    params += - learning_rate * grads

    return params
    # print('updated {}'.format(params))
