#!/usr/bin/python
__author__ = 'ehsanh'

import numpy as np
import os
from PIL import Image
import pickle
import cPickle
import random
import sys
import time
import maxpool3d
import theano
import theano.tensor as T
from theano.tensor import nnet
from theano.tensor.signal import downsample
import ipdb
# T = theano.tensor
# import theano.tensor.nnet.conv3d2d
import myconv3d2d_v2
from itertools import izip
from sklearn import preprocessing
import scipy.io as sio
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
    ACT_TANH = 't'
    ACT_SIGMOID = 's'
    ACT_ReLu = 'r'
    ACT_SoftPlus = 'p'

    def __init__(self, rng, input, signal_shape, filter_shape, poolsize=(2, 2, 2), stride=None, if_pool=False, act=None,
                 share_with=None,
                 tied=None, border_mode='valid'):
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

        #ipdb.set_trace()
        # Image padding
        # padded_out = downsample.max_pool_2d(
        #         input=input,
        #         ds=(1,1),
        #         padding=(filter_shape[-1]-1, filter_shape[-2]-1),
        #         ignore_border=True)
        # tmp = padded_out.transpose(0,2,3,4,1)
        # tmp_padded_out = downsample.max_pool_2d(
        #         input=tmp,
        #         ds=(1,1),
        #         padding=(0, filter_shape[1]-1),
        #         ignore_border=True)
        # padded_input = tmp_padded_out

        # convolution
        conv_out = myconv3d2d_v2.conv3d(
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
        else:
            tmp = conv_out + self.b.dimshuffle('x', 'x', 0, 'x', 'x')

        if act == ConvolutionLayer3D.ACT_TANH:
            self.output = T.tanh(tmp)
        elif act == ConvolutionLayer3D.ACT_SIGMOID:
            self.output = nnet.sigmoid(tmp)
        elif act == ConvolutionLayer3D.ACT_ReLu:
            self.output = tmp * (tmp>0)
        elif act == ConvolutionLayer3D.ACT_SoftPlus:
            self.output = T.log2(1+T.exp(tmp))
        else:
            self.output = tmp

        # store parameters of this layer
        self.params = [self.W, self.b]

        #EHA: parameter update- list of 5 previous updates
        # self.params_update = [5*[self.W_update], 5*[self.b_update]]

        self.deltas = [self.W_delta, self.b_delta]

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)


class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, share_with=None, activation=None):

        self.input = input

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



        # self.W = W
        # self.b = b

            self.W_delta = theano.shared(
                    np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    borrow=True
                )

            self.b_delta = theano.shared(value=b_values, borrow=True)

        self.params = [self.W, self.b]

        self.deltas = [self.W_delta, self.b_delta]

        lin_output = T.dot(self.input, self.W) + self.b

        # ipdb.set_trace()
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


class softmaxLayer(object):
    def __init__(self, input, n_in, n_out):

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

class CAE(object):
    def __init__(self, signal_shape, filter_shape, poolsize):
        rng = np.random.RandomState(None)
        dtensor5 = T.TensorType('float32', (False,)*5)
        inputs = dtensor5(name='inputs')
        # inputs_padded = dtensor5(name='inputs_padded')
        self.batchsize = signal_shape[0]
        self.in_channels   = signal_shape[2]
        self.in_time       = signal_shape[1]
        self.in_width      = signal_shape[4]
        self.in_height     = signal_shape[3]
        self.flt_channels  = filter_shape[0]
        self.flt_time      = filter_shape[1]
        self.flt_width     = filter_shape[4]
        self.flt_height    = filter_shape[3]

        self.hidden_layer=ConvolutionLayer3D(rng,
                                  input=inputs,
                                  signal_shape=signal_shape,
                                  filter_shape=filter_shape,
                                  act=ConvolutionLayer3D.ACT_SIGMOID,
                                  border_mode='full')

        self.hidden_image_shape = (self.batchsize,
                                   self.in_time,
                                   self.flt_channels,
                                   self.in_height+self.flt_height-1,
                                   self.in_width+self.flt_width-1)

        self.hidden_filter_shape = (self.in_channels, self.flt_time, self.flt_channels, self.flt_height,
                                    self.flt_width)

        self.recon_layer=ConvolutionLayer3D(rng,
                                 input=self.hidden_layer.output,
                                 signal_shape=self.hidden_image_shape,
                                 filter_shape=self.hidden_filter_shape,
                                 act=ConvolutionLayer3D.ACT_SIGMOID,
                                 # tied=hidden_layer,
                                 border_mode='valid')

        # recon_layer.W = hidden_layer.W
        # recon_layer.W = recon_layer.W.dimshuffle(1,0,2,3)

        self.layers = [self.hidden_layer, self.recon_layer]
        self.params = sum([layer.params for layer in self.layers], [])
        L=T.sum(T.pow(T.sub(self.recon_layer.output, inputs), 2), axis=0)
        cost = 0.5*T.mean(L)
        grads = T.grad(cost, self.params)

        # learning_rate = 0.1
        # updates = [(param_i, param_i-learning_rate*grad_i)
        #            for param_i, grad_i in zip(self.params, grads)]

        updates = adadelta_updates(self.params, grads, rho=0.95, eps=1e-6)

        self.train = theano.function(
        [inputs],
        cost,
        updates=updates,
        name = "train cae model"
        )

        self.activation = maxpool3d.max_pool_3d(
                input=self.hidden_layer.output.dimshuffle(0,2,1,3,4),
                ds=poolsize,
                ignore_border=True)
        self.activation = self.activation.dimshuffle(0,2,1,3,4)
        self.get_activation = theano.function(
            [inputs],
            self.activation,
            updates=None,
            name='get hidden activation')

    def save(self, filename):
        f = open(filename, 'w')
        pickle.dump(self.layers[0].get_state(), f, -1)
        f.close()

    def load(self, filename):
        f = open(filename)
        param = pickle.load(f)
        self.layers[0].set_state(param)
        f.close()
        print 'model loaded from', filename


def do_pretraining_cae(data, model, image_shape, batch_size=1, max_epoch=1):
    progress_report = 1
    save_interval = 1800
    num_subjects = data.shape[0]
    num_batches = num_subjects/batch_size
    last_save = time.time()
    # loss = 0
    epoch = 0
    # ipdb.set_trace()
    print 'training CAE'
    while True:
        try:
            start_time = time.time()
            # for epoch in xrange(max_epoch):
            loss = 0
            for batch in xrange(num_batches):
                #ipdb.set_trace()
                batch_data = data[batch*batch_size:(batch+1)*batch_size]
                batch_data = batch_data.reshape(image_shape)
                start=time.time()
                cost = model.train(batch_data)
                loss += cost
                train_time=time.time()-start
                print 'batch:', batch, ' cost:', cost, ' time:', train_time/60., 'min'
            epoch += 1
            if epoch % progress_report == 0:
                # loss /= progress_report
                print '%d\t%g' % (epoch, loss)
                sys.stdout.flush()
                # loss = 0
            if time.time() - last_save >= save_interval:
                # loss_history.append(loss)
                filename = 'cae_'+time.strftime('%Y%m%d-%H%M%S') + ('-%06d.pkl' % epoch)
                model.save(filename)
                print 'model saved to', filename
                last_save = time.time()
        except KeyboardInterrupt:
            filename = 'cae_'+time.strftime('%Y%m%d-%H%M%S') + ('-%06d.pkl' % epoch)
            model.save(filename)
            print 'model saved to', filename
            return filename

def get_all_hidden(data, model):
    hidden = model.get_activation(data)
    sio.savemat('hidden.mat',{'hidden':hidden})



def main():
    # rng = np.random.RandomState(23455)
    data = np.load('ADNI_npz/AD.npz')
    AD = data['AD']
    #ipdb.set_trace()
    # min_max_scaler = preprocessing.MinMaxScaler()
    # AD_scaled = min_max_scaler.fit_transform(AD)
    AD_scaled=(AD-AD.min(axis=(1,2,3)).reshape(AD.shape[0],1,1,1))\
              /(AD.max(axis=(1,2,3)).reshape(AD.shape[0],1,1,1))
    num_subjects, depth, height, width = AD.shape
    print 'data loaded and scaled'

     # define inputs and filters
    batchsize     = 1
    in_channels   = 1
    in_time       = depth
    in_width      = width
    in_height     = height
    flt_channels  = 8
    flt_time      = 3
    flt_width     = 3
    flt_height    = 3

    image_shp = (batchsize, in_time, in_channels, in_height, in_width)
    filter_shp = (flt_channels, flt_time, in_channels, flt_height, flt_width)

    cae1 = CAE(signal_shape=image_shp, filter_shape=filter_shp, poolsize=(2, 2, 2))
    print 'cae model built'
    do_pretraining_cae(AD_scaled, cae1, image_shape=image_shp, max_epoch=100)


    # ipdb.set_trace()

if __name__ == '__main__':
    sys.exit(main())

# class stacked_CAE(object):
#     def __init__(self, datafile=None):
#         #ipdb.set_trace()
#         rng = np.random.RandomState(None)
#         dtensor5 = T.TensorType('float32', (False,)*5)
#         inputs = dtensor5(name='inputs')
#
#         # define inputs and filters
#         batchsize     = 1
#         in_channels   = 1
#         in_time       = depth
#         in_width      = width
#         in_height     = height
#         flt_channels  = 1
#         flt_time      = 9
#         flt_width     = 9
#         flt_height    = 9
#         num_panes = [6, 16]
#         poolsize1=(3,3)
#         poolsize2=(2,2)
#         cae1_hidden_layer=ConvolutionLayer(rng,
#                                   input=shreds,
#                                   filter_shape=(num_panes[0], 1, 5, 5),
#                                   act=ConvolutionLayer.ACT_ReLu,
#                                   border_mode='full')
#
#         #filter_shape_hidden=(1, num_panes[0], 5, 5)
#
#         cae1_recon_layer=ConvolutionLayer(rng,
#                                  input=cae1_hidden_layer.output,
#                                  filter_shape=(1, num_panes[0], 5, 5),
#                                  act=ConvolutionLayer.ACT_ReLu,
#                                  border_mode='valid')
#
#         cae1_recon_layer.W = cae1_hidden_layer.W.dimshuffle(1,0,2,3)
#
#         self.params_cae1 = cae1_hidden_layer.params + cae1_recon_layer.params
#
#         L_cae1=T.sum(T.pow(T.sub(cae1_recon_layer.output, shreds), 2), axis=0)
#
#         cost_cae1 = 0.5*T.mean(L_cae1)
#
#         grads_cae1 = T.grad(cost_cae1, self.params_cae1)
#
#         learning_rate = 0.1
#         # updates_cae1 = [(param_i, param_i-learning_rate*grad_i)
#         #            for param_i, grad_i in zip(self.params_cae1, grads_cae1)]
#
#         updates_cae1 = adadelta_updates(self.params_cae1, grads_cae1, rho=0.95, eps=1e-6)
#
#         self.train_cae1 = theano.function(
#         [shreds],
#         cost_cae1,
#         updates=updates_cae1,
#         name = "train cae1 model"
#         )
#
#
#         cae1_activation = downsample.max_pool_2d(
#                 input=cae1_hidden_layer.output,
#                 ds=poolsize1,
#                 ignore_border=True)
#
#         self.cae1_get_activation = theano.function(
#             [shreds],
#             cae1_activation,
#             updates=None,
#             name='get hidden activation'
#         )
#
#         cae2_hidden_layer=ConvolutionLayer(rng,
#                                   input=cae1_activation,
#                                   filter_shape=(num_panes[1], num_panes[0], 5, 5),
#                                   act=ConvolutionLayer.ACT_ReLu,
#                                   border_mode='full')
#
#         cae2_recon_layer=ConvolutionLayer(rng,
#                                  input=cae2_hidden_layer.output,
#                                  filter_shape=(num_panes[0], num_panes[1], 5, 5),
#                                  act=ConvolutionLayer.ACT_ReLu,
#                                  border_mode='valid')
#
#         cae2_recon_layer.W = cae2_hidden_layer.W.dimshuffle(1,0,2,3)
#
#         self.params_cae2 = cae2_hidden_layer.params + cae2_recon_layer.params
#
#         L_cae2 = T.sum(T.pow(T.sub(cae2_recon_layer.output, cae1_activation),2), axis=0)
#
#         cost_cae2 = 0.5*T.mean(L_cae2)
#
#         grads_cae2 = T.grad(cost_cae2, self.params_cae2)
#
#         # updates_cae2 = [(param_i, param_i-learning_rate*grad_i)
#         #            for param_i, grad_i in zip(self.params_cae2, grads_cae2)]
#
#         updates_cae2 = adadelta_updates(self.params_cae2, grads_cae2, rho=0.95, eps=1e-6)
#
#         self.train_cae2 = theano.function(
#         [shreds],
#         cost_cae2,
#         updates=updates_cae2,
#         name = "train cae2 model"
#         )
#
#         self.layers = [cae1_hidden_layer, cae2_hidden_layer]
#         self.params = sum([l.params for l in self.layers], [])
#
#         cae2_activation = downsample.max_pool_2d(
#                 input=cae2_hidden_layer.output,
#                 ds=poolsize2,
#                 ignore_border=True)
#
#         cae3_hidden_layer=ConvolutionLayer(rng,
#                                   input=cae2_activation,
#                                   filter_shape=(NUM_OUTPUT, num_panes[1], 5, 5),
#                                   act=ConvolutionLayer.ACT_ReLu,
#                                   border_mode='full')
#
#         cae3_recon_layer=ConvolutionLayer(rng,
#                                  input=cae3_hidden_layer.output,
#                                  filter_shape=(num_panes[1], NUM_OUTPUT, 5, 5),
#                                  act=ConvolutionLayer.ACT_ReLu,
#                                  border_mode='valid')
#
#         cae3_recon_layer.W = cae3_hidden_layer.W.dimshuffle(1,0,2,3)
#
#         self.params_cae3 = cae3_hidden_layer.params + cae3_recon_layer.params
#
#         L_cae3 = T.sum(T.pow(T.sub(cae3_recon_layer.output, cae2_activation),2), axis=0)
#
#         cost_cae3 = 0.5*T.mean(L_cae3)
#
#         grads_cae3 = T.grad(cost_cae3, self.params_cae3)
#
#         # updates_cae3 = [(param_i, param_i-learning_rate*grad_i)
#         #            for param_i, grad_i in zip(self.params_cae3, grads_cae3)]
#
#         updates_cae3 = adadelta_updates(self.params_cae3, grads_cae3, rho=0.95, eps=1e-6)
#
#         self.train_cae3 = theano.function(
#         [shreds],
#         cost_cae3,
#         updates=updates_cae3,
#         name = "train cae3 model"
#         )
#
#         self.layers = [cae1_hidden_layer, cae2_hidden_layer, cae3_hidden_layer]
#         self.params = sum([l.params for l in self.layers], [])
#
#
#     def save(self, filename):
#         f = open(filename, 'w')
#         #ipdb.set_trace()
#         for l in self.layers:
#             pickle.dump(l.get_state(), f, -1)
#         f.close()
#     def load(self, filename):
#         f = open(filename)
#         for l in self.layers:
#             l.set_state(pickle.load(f))
#         f.close()
#         print 'model loaded from', filename


