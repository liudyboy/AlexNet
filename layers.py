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
import gc




class conv2d():
    def __init__(self, filters, kernel=[2, 2], use_bias=True, stride=(1, 1), padding=None, cover_all=False, activation='relu', name=None, normalization=None):
        """
          Args:
            filters: the number of filters, shape: one int number
            kernel: the kernel size, shape: [height, width]
            padding: can be string "SAME" means use 0 to padding the input array
            normalization: can be string 'local_response_normalization'
        """


        self.filters = filters
        self.kernel = kernel
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        self.cover_all = cover_all
        self.activation = activation
        self.normalization = normalization

        self.w = None
        self.b = None
        self.name = name


    def forward(self, x):
        """
          Args:
            x: shape [batch, channels, height, width]
        """

        # initial W and b
        # W shape: [out_channels, in_channels, kernel_height, kernel_width]
        if self.w is None:
            w = np.zeros(shape=(self.filters, x.shape[1], self.kernel[0], self.kernel[1]), dtype=np.float32)
            b = np.zeros(shape=(self.filters), dtype=np.float32)
            chainer.initializers.LeCunNormal()(w)

            self.w = chainer.as_variable(w)
            self.b = chainer.as_variable(b)


        self.x = chainer.as_variable(x)
        self.batch = ((x.shape[0]))

        self.pad = [0, 0]
        if self.padding == 'SAME':
            self.pad[0], self.pad[1] = int(np.ceil((self.w.shape[2] - 1) / 2)), int(np.ceil((self.w.shape[3] - 1) / 2))

        self.temp_result = F.convolution_2d(self.x, W = self.w, b = self.b, stride = self.stride, pad = self.pad)
        if self.activation == 'relu':
            self.result = F.relu(self.temp_result)
        else:
            self.result = self.temp_result


        if self.normalization == "local_response_normalization":
            self.normal_result = F.local_response_normalization(self.result)
            return self.normal_result
        
        return self.result

    def backward(self, d_out, update_method='vanilla'):
        """
          Args:
            d_out: shape need to be the same as the output of the convolution
        """
        if not (d_out.shape) == (self.result.shape):
            raise Exception('Layer: {} apply backward function the d_out shape: {} and the outputs of this layer shape: {} is not match!'.format(self.name, d_out.shape, self.result.shape))


        if self.normalization == "local_response_normalization":
            d_out = chainer.grad(outputs=[self.normal_result], inputs=[self.result], grad_outputs=[d_out])
            if isinstance(d_out, (list)):
                d_out = d_out[0]

        if self.activation is not None:
            dtemp = chainer.grad(outputs=[self.result], inputs=[self.temp_result], grad_outputs=[d_out])
            if isinstance(dtemp, (list)):
                dtemp = dtemp[0]

            dw, dbias, dx = chainer.grad(outputs=[self.temp_result], inputs=[self.w, self.b, self.x], grad_outputs=[dtemp])
        else: 
            dw, dbias, dx = chainer.grad(outputs=[self.result], inputs=[self.w, self.b, self.x], grad_outputs=[d_out])


        self.fbatch = Variable(np.array(self.batch, dtype=np.float32))
        if update_method == 'vanilla':
            update.vanilla_update(self.w, dw/self.fbatch)
            update.vanilla_update(self.b, dbias/self.fbatch)

            del d_out, dw, dbias, dtemp
        return dx



class max_pool2d():
    def __init__(self, ksize, stride=(1, 1), pad=0, cover_all=True, name=None):
        """
          Args:
            ksize: pair of ints, size of pooling window
            cover_all: bool, If True, all spatial locations are pooled into some output pixels. It may make the output size larger

        """
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.cover_all = cover_all
        self.name = name


    def forward(self, x):
        self.x = chainer.as_variable(x)
        self.result = F.max_pooling_2d(self.x, ksize = self.ksize, stride=self.stride, pad = self.pad, cover_all=self.cover_all)
        return self.result


    def backward(self, d_out):

        if not (d_out.shape) == (self.result.shape):
            raise Exception('Layer: {} apply backward function the d_out shape: {} and the outputs of this layer shape: {} is not match!'.format(self.name, d_out.shape, self.result.shape))

        dx = chainer.grad(outputs=[self.result], inputs=[self.x], grad_outputs=[d_out])

        if isinstance(dx, (list)):
            dx = dx[0]
        return dx

class dense():
    def __init__(self, out_size, activation=None, name=None, dropout=None):
        """
          Args:
            out_size: the output size of full connected layer
            W: shape [out_size, in_size]
            b: shape [out_size]

            dropout: bool flag, use the dropout method and default ratio=0.5
        """
        self.out_size = out_size
        self.w = None
        self.b = None
        self.activation = activation
        self.name = name
        self.dropout = dropout

    def forward(self, x):
        """
          Args:
            x: shape [batch, channels, height, width] or shape [batch, in_size]
        """
        self.init_x = chainer.as_variable(x)
        self.x = None
        self.batch = ((x.shape[0]))

        if len(x.shape) == 4:
            self.in_size = x.shape[1] * x.shape[2] * x.shape[3]
            self.x = chainer.as_variable(x.reshape(self.batch, -1))
        elif len(x.shape) == 2:
            self.in_size = x.shape[1]
            self.x = chainer.as_variable(x)
        else:
            raise Exception('Layer {}, input: {} in not legal'.format(self.name, x.shape))


        

        # initial W and b
        if self.w is None:
            w = np.zeros(shape=(self.out_size, self.in_size), dtype=np.float32)
            b = np.zeros(shape=(self.out_size), dtype=np.float32)
            chainer.initializers.LeCunNormal()(w)
            # chainer.initializers.LeCunNormal()(b)

            self.w = chainer.as_variable(w)
            self.b = chainer.as_variable(b)


        self.temp_result = F.linear(self.x, self.w, self.b)

        if self.activation == 'relu':
            self.result = F.relu(self.temp_result)
        else:
            self.result = self.temp_result

        if self.dropout is True:
            self.drop_result = F.dropout(self.result)
            return self.drop_result

        return self.result

    def backward(self, d_out, update_method='vanilla'):

        if not (d_out.shape) == (self.result.shape):
            raise Exception('Layer: {} apply backward function the d_out shape: {} and the outputs of this layer shape: {} is not match!'.format(self.name, d_out.shape, self.result.shape))

        
        if self.dropout is True:
            d_out = chainer.grad(outputs=[self.drop_result], inputs=[self.result], grad_outputs=[d_out])
            if isinstance(d_out, (list)):
                d_out = d_out[0]


        if self.activation is not None:
            dtemp = chainer.grad(outputs=[self.result], inputs=[self.temp_result], grad_outputs=[d_out])
            if isinstance(dtemp, (list)):
                dtemp = dtemp[0]

            dw, dbias, dx = chainer.grad(outputs=[self.temp_result], inputs=[self.w, self.b, self.x], grad_outputs=[dtemp])
            del dtemp

        else: 
            dw, dbias, dx = chainer.grad(outputs=[self.result], inputs=[self.w, self.b, self.x], grad_outputs=[d_out])

        


        self.fbatch = chainer.as_variable(np.array(self.batch, dtype=np.float32))
        if update_method == 'vanilla':
            
            update.vanilla_update(self.w, dw / self.fbatch)
            update.vanilla_update(self.b, dbias / self.fbatch)

        if not (self.init_x.shape == dx.shape):
            dx = dx.reshape(self.init_x.shape)

        del dw, dbias, d_out

        return dx
