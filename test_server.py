from rpyc import Service
from rpyc.utils.server import ThreadedServer
from threading import Thread
import rpyc
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
import layers_gpu as layers
import utils
import gc
import time
import numpy as np
import pickle
import sys



class Server(Service):


    def exposed_forward(self, input):

        ts1 = time.time()
        input = pickle.loads(input)
        # print(input[0])
        input = pickle.dumps(input)
        ts2 = time.time()

        print("cost time:", (ts2 - ts1) * 1000.)

        return input



if __name__ == '__main__':
    myserver= Server()
    s = ThreadedServer(myserver, port=18871, protocol_config=rpyc.core.protocol.DEFAULT_CONFIG)

    s.start()
