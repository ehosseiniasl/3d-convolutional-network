#!/usr/bin/python
"""
Stacked 3D-CAE for Alzheimer

11-11-15 Ehsan Hosseini-Asl

"""
__author__ = 'ehsanh'

import numpy as np
import argparse
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
import random
FLOAT_PRECISION = np.float32
from operator import eq

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
    # ACT_TANH = 't'
    # ACT_SIGMOID = 's'
    # ACT_ReLu = 'r'
    # ACT_SoftPlus = 'p'

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

class CAE3d(object):
    def __init__(self, signal_shape, filter_shape, poolsize, activation=None):
        rng = np.random.RandomState(None)
        dtensor5 = T.TensorType('float32', (False,)*5)
        self.inputs = dtensor5(name='inputs')
        # inputs_padded = dtensor5(name='inputs_padded')
        self.image_shape = signal_shape
        self.batchsize = signal_shape[0]
        self.in_channels   = signal_shape[2]
        self.in_depth       = signal_shape[1]
        self.in_width      = signal_shape[4]
        self.in_height     = signal_shape[3]
        self.flt_channels  = filter_shape[0]
        self.flt_time      = filter_shape[1]
        self.flt_width     = filter_shape[4]
        self.flt_height    = filter_shape[3]

        self.hidden_layer=ConvolutionLayer3D(rng,
                                  input=self.inputs,
                                  signal_shape=signal_shape,
                                  filter_shape=filter_shape,
                                  act=activation,
                                  border_mode='full')

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
                                 # tied=hidden_layer,
                                 border_mode='valid')

        # recon_layer.W = hidden_layer.W
        # recon_layer.W = recon_layer.W.dimshuffle(1,0,2,3)

        self.layers = [self.hidden_layer, self.recon_layer]
        self.params = sum([layer.params for layer in self.layers], [])
        # L=T.sum(T.pow(T.sub(self.recon_layer.output, self.inputs), 2), axis=0)
        L=T.sum(T.pow(T.sub(self.recon_layer.output, self.inputs), 2), axis=(1,2,3,4))
        self.cost = 0.5*T.mean(L)
        self.grads = T.grad(self.cost, self.params)

        # learning_rate = 0.1
        # updates = [(param_i, param_i-learning_rate*grad_i)
        #            for param_i, grad_i in zip(self.params, grads)]

        self.updates = adadelta_updates(self.params, self.grads, rho=0.95, eps=1e-6)

        self.train = theano.function(
        [self.inputs],
        self.cost,
        updates=self.updates,
        name = "train cae model"
        )
    # def train(self, self.inputs):
    #     train = theano.function(
    #     [self.inputs],
    #     self.cost,
    #     updates=self.updates,
    #     name = "train cae model"
    #     )

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
        self.flt_channels  = filter_shapes[0][0]
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

        conv1_output_shape = (self.batchsize,
                              self.in_depth/2,
                              self.flt_channels,
                              (self.in_height-self.flt_height+1)/2,
                              (self.in_width-self.flt_width+1)/2)

        conv2 = ConvolutionLayer3D(rng,
                                   input=conv1.output,
                                   signal_shape=conv1_output_shape,
                                   filter_shape=filter_shapes[1],
                                   act=activation_cae,
                                   poolsize=poolsize,
                                   if_pool=True,
                                   border_mode='valid')

        conv2_output_shape = (self.batchsize,
                              conv1_output_shape[1]/2,
                              self.flt_channels,
                              (conv1_output_shape[3]-self.flt_height+1)/2,
                              (conv1_output_shape[4]-self.flt_width+1)/2)

        conv3 = ConvolutionLayer3D(rng,
                                   input=conv2.output,
                                   signal_shape=conv2_output_shape,
                                   filter_shape=filter_shapes[2],
                                   act=activation_cae,
                                   poolsize=poolsize,
                                   if_pool=True,
                                   border_mode='valid')

        conv3_output_shape = (self.batchsize,
                              conv2_output_shape[1]/2,
                              self.flt_channels,
                              (conv2_output_shape[3]-self.flt_height+1)/2,
                              (conv2_output_shape[4]-self.flt_width+1)/2)

        # for layer in hidden_size:
        ip1_input=conv3.output.flatten(2)
        ip1 = HiddenLayer(rng,
                          input=ip1_input,
                          n_in=np.prod(conv3_output_shape[1:]),
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
                                    n_in=hidden_size[3],
                                    n_out=hidden_size[4])

        self.layers = [conv1,
                       conv2,
                       conv3,
                       ip1,
                       ip2,
                       # ip3,
                       # ip4,
                       output_layer]

        # freeze first 3 layers
        self.params = sum([l.params for l in self.layers[3:]], [])
        self.cost = output_layer.negative_log_likelihood(labels)
        self.grads = T.grad(self.cost, self.params)

        self.updates = adadelta_updates(parameters=self.params,
                                        gradients=self.grads,
                                        rho=0.95,
                                        eps=1e-6)

        self.error = output_layer.errors(labels)
        self.y_pred = output_layer.y_pred
        self.prob = output_layer.p_y_given_x.max(axis=1)
        self.train = theano.function(
            inputs=[images, labels],
            outputs=(self.error, self.cost, self.y_pred, self.prob),
            updates=self.updates
        )

        self.forward = theano.function(
            inputs=[images],
            outputs=self.layers[-1].y_pred
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


def load_batch(batch_idx, image_shape, data_dir, data_list):
    # data_list = os.listdir(data_dir)
    batch_size, d, c, h, w = image_shape
    batch_list = data_list[batch_idx*batch_size:(batch_idx+1)*batch_size]
    batch_data = np.empty(image_shape, dtype=FLOAT_PRECISION)
    labels = []
    for index, filename in enumerate(batch_list):
        data = sio.loadmat(data_dir+filename)
        data = data['original'].reshape(1, d, c, h, w)
        data_scaled = (data-data.min())/data.max()
        if 'AD' in filename:
            labels.append(0)
        elif 'MCI' in filename:
            labels.append(1)
        elif 'Normal' in filename:
            labels.append(2)
        else:
            raise NameError('filename is not ADNI subject')
        batch_data[index] = data_scaled
    return batch_data, labels

def do_pretraining_cae(data_dir, models, cae_layer, max_epoch=1):
    data_list = os.listdir(data_dir)
    random.shuffle(data_list)
    # data_list = shuffled_list
    batch_size, d, c, h, w = models[cae_layer-1].image_shape
    progress_report = 1
    save_interval = 1800
    # num_subjects = data.shape[0]
    num_subjects = len(data_list)
    num_batches = num_subjects/batch_size
    last_save = time.time()
    # loss = 0
    epoch = 0
    print 'training CAE_'+str(cae_layer)
    while True:
        try:
            # for epoch in xrange(max_epoch):
            loss = 0
            start_time = time.time()
            for batch in xrange(num_batches):
                # ipdb.set_trace()
                batch_data, labels = load_batch(batch, models[0].image_shape, data_dir, data_list=data_list)
                # batch_data = data[batch*batch_size:(batch+1)*batch_size]
                # batch_data = batch_data.reshape(image_shape)
                start = time.time()
                if cae_layer == 1:
                    cost = models[0].train(batch_data)
                elif cae_layer == 2:
                    hidden_batch = models[0].get_activation(batch_data)
                    cost = models[1].train(hidden_batch)
                elif cae_layer == 3:
                    hidden_batch1 = models[0].get_activation(batch_data)
                    hidden_batch2 = models[1].get_activation(hidden_batch1)
                    cost = models[2].train(hidden_batch2)
                loss += cost
                train_time = time.time()-start
                # print 'batch:', batch, ' cost:', cost, ' time:', train_time/60., 'min'
                print 'batch:%d\tcost:%.2f\ttime:%.2f' % (batch, cost, train_time/60.)
                sys.stdout.flush()
            epoch += 1
            if epoch % progress_report == 0:
                # loss /= progress_report
                print '%d\t%g\t%f' % (epoch, loss, time.time()-start_time)
                sys.stdout.flush()
                # loss = 0
            if time.time() - last_save >= save_interval:
                # loss_history.append(loss)
                filename = 'cae'+str(cae_layer)+'_'+time.strftime('%Y%m%d-%H%M%S') + ('-%06d.pkl' % epoch)
                models[cae_layer-1].save(filename)
                print 'model saved to', filename
                sys.stdout.flush()
                last_save = time.time()
            if epoch >= max_epoch-1:
                filename = 'cae'+str(cae_layer)+'_'+time.strftime('%Y%m%d-%H%M%S') + ('-%06d.pkl' % epoch)
                models[cae_layer-1].save(filename)
                print 'max epoch reached. model saved to', filename
                sys.stdout.flush()
                return filename

        except KeyboardInterrupt:
            filename = 'cae_'+time.strftime('%Y%m%d-%H%M%S') + ('-%06d.pkl' % epoch)
            models[cae_layer-1].save(filename)
            print 'model saved to', filename
            sys.stdout.flush()
            return filename

def finetune_scae(data_dir, model, max_epoch=1):
    data_list = os.listdir(data_dir)
    random.shuffle(data_list)
    batch_size, d, c, h, w = model.image_shape
    progress_report = 1
    save_interval = 1800
    num_subjects = len(data_list)
    num_batches = num_subjects/batch_size
    last_save = time.time()
    epoch = 0
    print 'training scae'
    while True:
        try:
            loss_hist = np.empty((num_batches,), dtype=FLOAT_PRECISION)
            error_hist = np.empty((num_batches,), dtype=FLOAT_PRECISION)
            start_time = time.time()
            for batch in xrange(num_batches):
                batch_data, batch_labels = load_batch(batch, model.image_shape, data_dir, data_list=data_list)
                start = time.time()
                # ipdb.set_trace()
                batch_error, cost, pred, prob = model.train(batch_data, batch_labels)
                loss_hist[batch] = cost
                train_time = time.time()-start
                print
                # print 'batch:', batch, ' cost:', cost, ' time:', train_time/60., 'min'
                # batch_error = (map(eq, batch_labels, pred.tolist()).index(False))/float(batch_size)
                error_hist[batch] = batch_error
                print 'batch:%d\terror:%.2f\tcost:%.2f\ttime:%.2f' % (batch, batch_error, cost, train_time/60.)
                # print 'labels:   ', np.asarray(batch_labels)
                print 'labels:\t',
                for l in batch_labels:
                    print l,
                print
                print 'pred:\t',
                for p in pred:
                    print p,
                print
                print 'prob:',
                for p in prob:
                    print '%.2f'%p,
                print
                # print 'probability:', np.float16(prob)
                sys.stdout.flush()
            epoch += 1
            if epoch % progress_report == 0:
                print 'epoch:%d\terror:%g\tloss:%g\ttime:%f' % (epoch, error_hist.mean(), loss_hist.mean(), time.time()-start_time)
                sys.stdout.flush()
            if time.time() - last_save >= save_interval:
                # filename = 'scae_'+time.strftime('%Y%m%d-%H%M%S') + ('-%06d.pkl' % epoch)
                filename = 'scae.pkl'
                model.save(filename)
                print 'scae model saved to', filename
                sys.stdout.flush()
                last_save = time.time()
            if epoch >= max_epoch-1:
                # filename = 'scae_'+time.strftime('%Y%m%d-%H%M%S') + ('-%06d.pkl' % epoch)
                filename = 'scae.pkl'
                model.save(filename)
                print 'max epoch reached. scae model saved to', filename
                sys.stdout.flush()
                return filename
        except KeyboardInterrupt:
            # filename = 'scae_'+time.strftime('%Y%m%d-%H%M%S') + ('-%06d.pkl' % epoch)
            filename = 'scae.pkl'
            model.save(filename)
            print 'scae model saved to', filename
            sys.stdout.flush()
            return filename


def get_hidden_data(dir, image_shape, model):
    # print 'get hidden activation'
    hidden_data = {}
    for type in ['AD', 'MCI', 'Normal']:
        print 'get %s hidden activation' % type
        data_dir = dir+type+'/'
        data_list = os.listdir(data_dir)
        data_list.sort()
        num_subjects = len(data_list)
        batch_size = image_shape[0]
        num_batches = num_subjects/batch_size

        sample_batch = load_batch(0, image_shape, data_dir, data_list=data_list)
        sample_hidden = model.get_activation(sample_batch)
        sample_shape = sample_hidden.shape
        _, depth, channel, height, width = sample_shape
        hidden_shape = (len(data_list), depth, channel, height, width)

        hidden_data[type] = np.empty(hidden_shape, dtype=FLOAT_PRECISION)
        for batch in xrange(num_batches):
            print batch
            batch_data = load_batch(batch, image_shape, data_dir, data_list=data_list)
            batch_hidden = model.get_activation(batch_data)
            hidden_data[type][batch*batch_size:(batch+1)*batch_size] = batch_hidden
        for i in xrange(10):
            filename = type+'_hidden_data_'+str(i)+'.mat'
            sio.savemat(filename, {'hidden_data':hidden_data[type][i*10:(i+1)*10]})
    return hidden_data

def ProcessCommandLine():
    parser = argparse.ArgumentParser(description='train scae on alzheimer')
    default_image_dir = 'ADNI_original/data/'
    parser.add_argument('-I', '--data_dir', default=default_image_dir,
                        help='location of image files; default=%s' % default_image_dir)
    # parser.add_argument('-m', '--model_name',
    #                     help='start with this model')
    parser.add_argument('-cae1', '--cae1_model',
                        help='Initialize cae1 model')
    parser.add_argument('-cae2', '--cae2_model',
                        help='Initialize cae2 model')
    parser.add_argument('-cae3', '--cae3_model',
                        help='Initialize cae3 model')
    parser.add_argument('-ac', '--activation_cae', type=str, default='relu',
                        help='cae activation function')
    parser.add_argument('-af', '--activation_final', type=str, default='relu',
                        help='final layer activation function')
    parser.add_argument('-fn', '--filter_channel', type=int, default=8,
                        help='filter channel')
    parser.add_argument('-fs', '--filter_size', type=int, default=3,
                        help='filter size')
    # parser.add_argument('-p', '--pretrain', action='store_true',
    #                      help='do pretraining')
    parser.add_argument('-p', '--pretrain_layer', type=int, default=0,
                        help='pretrain cae layer')
    parser.add_argument('-t', '--test', action='store_true',
                        help='do testing')
    parser.add_argument('-ft', '--finetune', action='store_true',
                        help='do fine tuning')
    parser.add_argument('-batch', '--batchsize', type=int, default=1,
                        help='batch size')
    args = parser.parse_args()
    return args.data_dir, args.cae1_model, args.cae2_model, args.cae3_model, args.activation_cae, args.activation_final, \
           args.filter_channel, args.filter_size, args.pretrain_layer, args.test, \
           args.finetune, args.batchsize



def main():
    # AD_dir = 'ADNI_original/AD/'
    # MCI_dir = 'ADNI_original/MCI/'
    # Normal_dir = 'ADNI_original/Normal/'
    dir = 'ADNI_original/'
    # data_dir = dir+'data/'
    data_dir, cae1_model, cae2_model, cae3_model, activation_cae, activation_final, \
    flt_channels, flt_size, pretrain_layer, test, finetune, batchsize = ProcessCommandLine()
    print 'cae activation:', activation_cae
    print 'final layers activation:', activation_final
    data_list = os.listdir(data_dir)
    sample = sio.loadmat(data_dir+data_list[0])
    depth, height, width = sample['original'].shape
    # num_subjects = len(data_list)

     # define inputs and filters
    # batchsize     = 5
    in_channels   = 1
    in_time       = depth
    in_width      = width
    in_height     = height
    # flt_channels  = 8
    flt_depth     = flt_size
    flt_width     = flt_size
    flt_height    = flt_size

    image_shp = (batchsize, in_time, in_channels, in_height, in_width)
    filter_shp_1 = (flt_channels, flt_depth, in_channels, flt_height, flt_width)
    filter_shp_2 = (flt_channels, flt_depth, filter_shp_1[0], flt_height, flt_width)
    filter_shp_3 = (flt_channels, flt_depth, filter_shp_2[0], flt_height, flt_width)

    # print 'building CAE'
    # cae1 = CAE3d(signal_shape=image_shp, filter_shape=filter_shp, poolsize=(2, 2, 2))
    # print 'model built'
    # sys.stdout.flush()
    # if model_name:
    #     cae1.load(model_name)

    if not finetune:
        cae1 = CAE3d(signal_shape=image_shp,
                     filter_shape=filter_shp_1,
                     poolsize=(2, 2, 2),
                     activation=activation_cae)
        print 'CAE1 built'
        if cae1_model:
            cae1.load(cae1_model)
            # print 'CAE1 loaded by: ', cae1_model
        sys.stdout.flush()

        cae2 = CAE3d(signal_shape=cae1.hidden_pooled_image_shape,
                     filter_shape=filter_shp_2,
                     poolsize=(2, 2, 2),
                     activation=activation_cae)
        print 'CAE2 built'
        if cae2_model:
            cae2.load(cae2_model)
            # print 'CAE2 loaded by: ', cae2_model
        sys.stdout.flush()

        cae3 = CAE3d(signal_shape=cae2.hidden_pooled_image_shape,
                     filter_shape=filter_shp_3,
                     poolsize=(2, 2, 2),
                     activation=activation_cae)
        print 'CAE3 built'
        if cae3_model:
            cae3.load(cae3_model)
            # print 'CAE3 loaded by: ', cae3_model
        sys.stdout.flush()

        # if test:
        #     cae1_hidden = get_hidden_data(dir=dir, image_shape=image_shp, model=cae1)
        #     cae2_hidden = get_hidden_data(dir=dir, image_shape=image_shp, model=cae1)
        #     cae3_hidden = get_hidden_data(dir=dir, image_shape=image_shp, model=cae1)
        if pretrain_layer != 0:
            cae_models = [cae1, cae2, cae3]
            do_pretraining_cae(data_dir=data_dir,
                               models=cae_models,
                               cae_layer=pretrain_layer,
                               max_epoch=100)
    elif finetune:
        print 'creating scae...'
        scae = stacked_CAE3d(image_shape=image_shp,
                             filter_shapes=(filter_shp_1, filter_shp_2, filter_shp_3),
                             poolsize=(2, 2, 2),
                             activation_cae=activation_cae,
                             activation_final=activation_final,
                             hidden_size=(2000, 500, 200, 20, 3))
        print 'scae model built'
        if cae1_model:
            scae.load_cae(cae1_model, cae_layer=0)
            pass
        if cae2_model:
            scae.load_cae(cae2_model, cae_layer=1)
            pass
        if cae3_model:
            scae.load_cae(cae3_model, cae_layer=2)
            pass

        finetune_scae(data_dir=data_dir,
                      model=scae,
                      max_epoch=100)

    # elif pretrain == 2:
    #     do_pretraining_cae(data_dir=data_dir,
    #                        models=cae_models,
    #                        cae_layer=pretrain,
    #                        image_shape=cae1.hidden_pooled_image_shape,
    #                        max_epoch=100)
    # elif pretrain == 3:
    #     do_pretraining_cae(data_dir=data_dir,
    #                        models=cae_models,
    #                        cae_layer=pretrain,
    #                        image_shape=cae2.hidden_pooled_image_shape,
    #                        max_epoch=100)
    # elif test:
    #     get_hidden_data(dir=dir, image_shape=image_shp, model=cae1)


    # ipdb.set_trace()

if __name__ == '__main__':
    sys.exit(main())

