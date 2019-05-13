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



def vanilla_update(params, grads, learning_rate=0.0001):
    """
    Args:
       params: parameters needed update by gradients, and this must be chainer Variable type
       grads: parameters gradinets, and this must be chainer Variable type
    """

    params.array += (- learning_rate * grads.array)

