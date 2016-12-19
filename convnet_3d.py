#!/usr/bin/python
"""
3D-CAE with max-pooling

Stacked 3D-CAE for Alzheimer

11-11-15 Ehsan Hosseini-Asl

"""
__author__ = 'ehsanh'

import numpy as np
import pickle
import maxpool3d
import theano
import theano.tensor as T
from theano.tensor import nnet
from theano.tensor.signal import downsample
import conv3d2d
from itertools import izip

FLOAT_PRECISION = np.float32

def adadelta_updates(parameters, gradients, rho, eps):

    # create variables to store intermediate updates
    # ipdb.set_trace()
    gradients_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=FLOAT_PRECISION),) for p in parameters ]
    deltas_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=FLOAT_PRECISION)) for p in parameters ]

    # calculates the new "average" delta for the next iteration
    gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in izip(gradients_sq,gradients) ]

    # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
    deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients) ]

    # calculates the new "average" deltas for the next step.
    deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas) ]

    # Prepare it as a list f
    gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
    deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
    parameters_updates = [ (p,p - d) for p,d in izip(parameters,deltas) ]
    # ipdb.set_trace()
    return gradient_sq_updates + deltas_sq_updates + parameters_updates
    # return parameters_updates


class ConvolutionLayer3D(object):

    def __init__(self, rng, input, signal_shape, filter_shape, poolsize=(2, 2, 2), stride=None, if_pool=False, if_hidden_pool=False,
                 act=None,
                 share_with=None,
                 tied=None,
                 border_mode='valid'):
        self.input = input

        if share_with:
            self.W = share_with.W
            self.b = share_with.b

            self.W_delta = share_with.W_delta
            self.b_delta = share_with.b_delta

        elif tied:
            self.W = tied.W.dimshuffle(1,0,2,3)
            self.b = tied.b

            self.W_delta = tied.W_delta.dimshuffle(1,0,2,3)
            self.b_delta = tied.b_delta

        else:
            fan_in = np.prod(filter_shape[1:])
            poolsize_size = np.prod(poolsize) if poolsize else 1
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / poolsize_size)
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)

            self.W_delta = theano.shared(
                np.zeros(filter_shape, dtype=theano.config.floatX),
                borrow=True
            )

            self.b_delta = theano.shared(value=b_values, borrow=True)

        # convolution
        conv_out = conv3d2d.conv3d(
            signals=input,
            filters=self.W,
            signals_shape=signal_shape,
            filters_shape=filter_shape,
            border_mode=border_mode)

        #if poolsize:
        if if_pool:
            conv_out = conv_out.dimshuffle(0,2,1,3,4) #maxpool3d works on last 3 dimesnions
            pooled_out = maxpool3d.max_pool_3d(
                input=conv_out,
                ds=poolsize,
                ignore_border=True)
            tmp_out = pooled_out.dimshuffle(0,2,1,3,4)
            tmp = tmp_out + self.b.dimshuffle('x', 'x', 0, 'x', 'x')
        elif if_hidden_pool:
            pooled_out = downsample.max_pool_2d(
                input=conv_out,
                ds=poolsize[:2],
                st=stride,
                ignore_border=True)
            tmp = pooled_out + self.b.dimshuffle('x', 'x', 0, 'x', 'x')
        else:
            tmp = conv_out + self.b.dimshuffle('x', 'x', 0, 'x', 'x')

        if act == 'tanh':
            self.output = T.tanh(tmp)
        elif act == 'sigmoid':
            self.output = nnet.sigmoid(tmp)
        elif act == 'relu':
            # self.output = tmp * (tmp>0)
            self.output = 0.5 * (tmp + abs(tmp)) + 1e-9
        elif act == 'softplus':
            # self.output = T.log2(1+T.exp(tmp))
            self.output = nnet.softplus(tmp)
        else:
            self.output = tmp

        self.get_activation = theano.function(
            [self.input],
            self.output,
            updates=None,
            name='get hidden activation')

        # store parameters of this layer
        self.params = [self.W, self.b]
        self.deltas = [self.W_delta, self.b_delta]

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)


class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, share_with=None, activation=None):

        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        if share_with:
            self.W = share_with.W
            self.b = share_with.b

            self.W_delta = share_with.W_delta
            self.b_delta = share_with.b_delta
        else:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == nnet.sigmoid:
                W_values *= 4

            self.W = theano.shared(value=W_values, name='W', borrow=True)

            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)

            self.W_delta = theano.shared(
                    np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    borrow=True
                )

            self.b_delta = theano.shared(value=b_values, borrow=True)

        self.params = [self.W, self.b]

        self.deltas = [self.W_delta, self.b_delta]

        lin_output = T.dot(self.input, self.W) + self.b

        if activation == 'tanh':
            self.output = T.tanh(lin_output)
        elif activation == 'sigmoid':
            self.output = nnet.sigmoid(lin_output)
        elif activation == 'relu':
            self.output = T.maximum(lin_output, 0)
        else:
            self.output = lin_output

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)

    def initialize_layer(self):
        rng = np.random.RandomState(None)
        W_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (self.n_in + self.n_out)),
                high=np.sqrt(6. / (self.n_in + self.n_out)),
                size=(self.n_in, self.n_out)),
            dtype=theano.config.floatX)

        if self.activation == nnet.sigmoid:
            W_values *= 4

        b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
        self.W.set_value(W_values, borrow=True)
        self.b.set_value(b_values, borrow=True)


class softmaxLayer(object):
    def __init__(self, input, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.W = theano.shared(
            value=np.zeros(
                (n_in,n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.W_delta = theano.shared(
                np.zeros((n_in,n_out), dtype=theano.config.floatX),
                borrow=True
            )

        self.b_delta = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX),
            name='b',
            borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

        self.deltas = [self.W_delta, self.b_delta]

    def initialize_layer(self):

        W_value=np.zeros(
            (self.n_in, self.n_out),
            dtype=theano.config.floatX
        )

        b_value=np.zeros(
            (self.n_out,),
            dtype=theano.config.floatX
        )

        self.W.set_value(W_value, borrow=True)
        self.b.set_value(b_value, borrow=True)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)


class CAE3d(object):
    def __init__(self, signal_shape, filter_shape, poolsize, activation=None):
        rng = np.random.RandomState(None)
        dtensor5 = T.TensorType('float32', (False,)*5)
        self.inputs = dtensor5(name='inputs')
        self.image_shape = signal_shape
        self.batchsize = signal_shape[0]
        self.in_channels   = signal_shape[2]
        self.in_depth      = signal_shape[1]
        self.in_width      = signal_shape[4]
        self.in_height     = signal_shape[3]
        self.flt_channels  = filter_shape[0]
        self.flt_time      = filter_shape[1]
        self.flt_width     = filter_shape[4]
        self.flt_height    = filter_shape[3]
        self.activation = activation

        self.hidden_layer=ConvolutionLayer3D(rng,
                                             input=self.inputs,
                                             signal_shape=signal_shape,
                                             filter_shape=filter_shape,
                                             act=activation,
                                             border_mode='full',
                                             if_hidden_pool=False)

        self.hidden_image_shape = (self.batchsize,
                                   self.in_depth,
                                   self.flt_channels,
                                   self.in_height+self.flt_height-1,
                                   self.in_width+self.flt_width-1)

        self.hidden_pooled_image_shape = (self.batchsize,
                                          self.in_depth/2,
                                          self.flt_channels,
                                          (self.in_height+self.flt_height-1)/2,
                                          (self.in_width+self.flt_width-1)/2)

        self.hidden_filter_shape = (self.in_channels, self.flt_time, self.flt_channels, self.flt_height,
                                    self.flt_width)

        self.recon_layer=ConvolutionLayer3D(rng,
                                 input=self.hidden_layer.output,
                                 signal_shape=self.hidden_image_shape,
                                 filter_shape=self.hidden_filter_shape,
                                 act=activation,
                                 border_mode='valid')

        self.layers = [self.hidden_layer, self.recon_layer]
        self.params = sum([layer.params for layer in self.layers], [])
        L=T.sum(T.pow(T.sub(self.recon_layer.output, self.inputs), 2), axis=(1,2,3,4))
        self.cost = 0.5*T.mean(L)
        self.grads = T.grad(self.cost, self.params)
        self.updates = adadelta_updates(self.params, self.grads, rho=0.95, eps=1e-6)

        self.train = theano.function(
        [self.inputs],
        self.cost,
        updates=self.updates,
        name = "train cae model"
        )

        self.activation = maxpool3d.max_pool_3d(
                input=self.hidden_layer.output.dimshuffle(0,2,1,3,4),
                ds=poolsize,
                ignore_border=True)
        self.activation = self.activation.dimshuffle(0,2,1,3,4)
        self.get_activation = theano.function(
            [self.inputs],
            self.activation,
            updates=None,
            name='get hidden activation')

    def save(self, filename):
        f = open(filename, 'w')
        for layer in self.layers:
            pickle.dump(layer.get_state(), f, -1)
        f.close()


    def load(self, filename):
        f = open(filename)
        for layer in self.layers:
            layer.set_state(pickle.load(f))
        f.close()
        print 'cae model loaded from', filename


class stacked_CAE3d(object):
    def __init__(self, image_shape, filter_shapes, poolsize, activation_cae=None, activation_final=None, hidden_size=(2000, 500, 200, 20, 3)):
        rng = np.random.RandomState(None)
        dtensor5 = T.TensorType('float32', (False,)*5)
        images = dtensor5(name='images')
        labels = T.lvector('labels')

        self.image_shape = image_shape
        self.batchsize = image_shape[0]
        self.in_channels   = image_shape[2]
        self.in_depth       = image_shape[1]
        self.in_width      = image_shape[4]
        self.in_height     = image_shape[3]
        self.flt_channels1  = filter_shapes[0][0]
        self.flt_channels2  = filter_shapes[1][0]
        self.flt_channels3  = filter_shapes[2][0]
        self.flt_time      = filter_shapes[0][1]
        self.flt_width     = filter_shapes[0][4]
        self.flt_height    = filter_shapes[0][3]

        conv1 = ConvolutionLayer3D(rng,
                                   input=images,
                                   signal_shape=image_shape,
                                   filter_shape=filter_shapes[0],
                                   act=activation_cae,
                                   poolsize=poolsize,
                                   if_pool=True,
                                   border_mode='valid')

        self.conv1_output_shape = (self.batchsize,
                              self.in_depth/2,
                              self.flt_channels1,
                              (self.in_height-self.flt_height+1)/2,
                              (self.in_width-self.flt_width+1)/2)

        #conv2_input=conv1.output.flatten(2)
        conv2 = ConvolutionLayer3D(rng,
                                   input=conv1.output,
                                   signal_shape=self.conv1_output_shape,
                                   filter_shape=filter_shapes[1],
                                   act=activation_cae,
                                   poolsize=poolsize,
                                   if_pool=True,
                                   border_mode='valid')

        self.conv2_output_shape = (self.batchsize,
                              self.conv1_output_shape[1]/2,
                              self.flt_channels2,
                              (self.conv1_output_shape[3]-self.flt_height+1)/2,
                              (self.conv1_output_shape[4]-self.flt_width+1)/2)

        conv3_input=conv2.output.flatten(2)
        conv3 = ConvolutionLayer3D(rng,
                                   input=conv2.output,
                                   signal_shape=self.conv2_output_shape,
                                   filter_shape=filter_shapes[2],
                                   act=activation_cae,
                                   poolsize=poolsize,
                                   if_pool=True,
                                   border_mode='valid')

        self.conv3_output_shape = (self.batchsize,
                              self.conv2_output_shape[1]/2,
                              self.flt_channels3,
                              (self.conv2_output_shape[3]-self.flt_height+1)/2,
                              (self.conv2_output_shape[4]-self.flt_width+1)/2)

        # 4 layers in hidden_size:
        ip1_input=conv3.output.flatten(2)
        ip1 = HiddenLayer(rng,
                          input=ip1_input,
                          n_in=np.prod(self.conv3_output_shape[1:]),
                          n_out=hidden_size[0],
                          activation=activation_final)

        ip2 = HiddenLayer(rng,
                          input=ip1.output,
                           n_in=hidden_size[0],
                           n_out=hidden_size[1],
                           activation=activation_final)

        ip3 = HiddenLayer(rng,
                          input=ip2.output,
                           n_in=hidden_size[1],
                           n_out=hidden_size[2],
                           activation=activation_final)

        ip4 = HiddenLayer(rng,
                          input=ip3.output,
                           n_in=hidden_size[2],
                           n_out=hidden_size[3],
                           activation=activation_final)

        output_layer = softmaxLayer(input=ip4.output,
                                    n_in=hidden_size[1],
                                    n_out=hidden_size[4])

        self.layers = [conv1,
                       conv2,
                       conv3,
                       ip1,
                       ip2,
                       output_layer]

        # freeze first 3 conv layers
        self.params = sum([l.params for l in self.layers[3:]], [])
        self.cost = output_layer.negative_log_likelihood(labels)
        self.grads = T.grad(self.cost, self.params)
        self.grads_input = T.grad(self.cost, images)

        self.updates = adadelta_updates(parameters=self.params,
                                        gradients=self.grads,
                                        rho=0.95,
                                        eps=1e-6)

        self.error = output_layer.errors(labels)
        self.y_pred = output_layer.y_pred
        self.prob = output_layer.p_y_given_x.max(axis=1)
        self.true_prob = output_layer.p_y_given_x[T.arange(labels.shape[0]), labels]
        self.p_y_given_x = output_layer.p_y_given_x
        self.train = theano.function(
            inputs=[images, labels],
            outputs=(self.error, self.cost, self.y_pred, self.prob),
            updates=self.updates
        )

        self.forward = theano.function(
            inputs=[images, labels],
            outputs=(self.error, self.y_pred, self.prob, self.true_prob, self.p_y_given_x,
                     conv3_input, ip1_input, self.layers[-2].output, self.layers[-3].output,
                     self.grads_input)
        )

    def load_cae(self, filename, cae_layer):
        f = open(filename)
        self.layers[cae_layer].set_state(pickle.load(f))
        print 'cae %d loaded from %s' % (cae_layer, filename)

    def save(self, filename):
        f = open(filename, 'w')
        for l in self.layers:
            pickle.dump(l.get_state(), f, -1)
        f.close()

    def load(self, filename):
        f = open(filename)
        for l in self.layers:
            l.set_state(pickle.load(f))
        f.close()
        print 'model loaded from', filename

    def load_binary(self, filename):
        f = open(filename)
        for l in self.layers[:-1]:
            l.set_state(pickle.load(f))
        f.close()
        print 'model loaded from', filename

    def load_conv(self, filename):
        f = open(filename)
        for l in self.layers[:3]:
            l.set_state(pickle.load(f))
        f.close()
        print 'model conv layers loaded from', filename

    def load_fc(self, filename):
        f = open(filename)
        for l in self.layers[-3:]:
            l.set_state(pickle.load(f))
        f.close()
        print 'model fc layers loaded from', filename


