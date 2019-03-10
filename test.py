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


x = chainer.as_variable(np.ones(shape=(128, 10, 10, 10), dtype=np.float32))
w = chainer.as_variable(np.ones(shape=(20, 10, 2, 2), dtype=np.float32))
b = chainer.as_variable(np.ones(shape=(20), dtype=np.float32))

x.to_gpu(0)
w.to_gpu(0)
b.to_gpu(0)
xp = cuda.get_array_module(x)
print("xp device:", xp)

# y = x + 1
# stride = chainer.as_variable(np.array([1, 1]))
# pad = chainer.as_variable(np.array([0, 0]))
# stride.to_gpu(0)
# pad.to_gpu(0)
stride = [1, 1]
pad = [0, 0]
y = F.convolution_2d(x, w, b, stride=stride, pad=pad)
# y = F.convolution_2d(x, w, b)

grad_out = chainer.as_variable(np.ones(shape=y.shape, dtype=np.float32))
grad_out.to_gpu(0)

ddw, db, dx = chainer.grad(outputs=[y], inputs=[w, b, x], grad_outputs=[grad_out])

update.vanilla_update(w, ddw)
print('dw', ddw)
