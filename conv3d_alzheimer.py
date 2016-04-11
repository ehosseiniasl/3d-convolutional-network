__author__ = 'ehsanh'

import theano
from theano import tensor as T
from theano.tensor.nnet import conv, Conv3D, conv3d2d
import numpy as np
import pylab
from PIL import Image
import ipdb
import matplotlib.pyplot as plt
import theano.sandbox.cuda.fftconv as fftconv
# T = theano.tensor
# floatX = theano.config.floatX
# import theano.tensor.nnet.conv3d2d

rng = np.random.RandomState(23455)

data = np.load('ADNI_npz/AD.npz')
AD = data['AD']
n_im, depth, height, width = AD.shape


# define inputs and filters
batchsize     = 1
in_channels   = 1
in_time       = depth
in_width      = width
in_height     = height
flt_channels  = 1
flt_time      = 9
flt_width     = 9
flt_height    = 9

# instantiate 4D tensor for input
dtensor5 = T.TensorType('float32', (False,)*5)
input = dtensor5(name='input')

# input = T.tensor4(name='input')

# initialize shared variable for weights.
# w_shp = (2, 1, 9, 9)
# w_shp = (2, 1, 1, 9, 9)
w_shp = (flt_channels, flt_time, in_channels, flt_height, flt_width)
w_bound = np.sqrt(1 * 9 * 9)
W = theano.shared( np.asarray(
            rng.uniform(
                low=-1.0 / w_bound,
                high=1.0 / w_bound,
                size=w_shp),
            dtype=input.dtype), name ='W')

# initialize shared variable for bias (1D tensor) with random values
# IMPORTANT: biases are usually initialized to zero. However in this
# particular application, we simply apply the convolutional layer to
# an image without learning the parameters. We therefore initialize
# them to random values to "simulate" learning.
b_shp = (2,)
b = theano.shared(np.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=input.dtype), name ='b')

# build symbolic expression that computes the convolution of input with filters in w
# conv_out = conv.conv2d(input, W)
# conv_out = conv3d2d.conv3d(input, W)
# conv_out = Conv3D(input, W)
# conv_out = conv3d_fft(input, W)

# outputA = T.nnet.conv3d2d.conv3d(
# 	signals=input,  # Ns, Ts, C, Hs, Ws
# 	filters=W, # Nf, Tf, C, Hf, Wf
# 	signals_shape=(batchsize, in_time, in_channels, in_height, in_width),
# 	filters_shape=(flt_channels, flt_time, in_channels, flt_height, flt_width),
# 	border_mode='valid')
# outputA = outputA + b.dimshuffle('x','x',0,'x','x')

outputB = fftconv.conv3d_fft(
	input=input,  # b, ic, i0, i1, i2
	filters=W, # oc, ic, f0, f1, f2
	image_shape=(batchsize, in_time, in_channels, in_height, in_width),
	filter_shape=(flt_channels, flt_time, in_channels, flt_height, flt_width),
	border_mode='full')
outputB = outputB + b.dimshuffle('x','x',0,'x','x')

# create theano function to compute filtered images
# fA = theano.function([input], outputA)
fB = theano.function([input], outputB)


ipdb.set_trace()
# open random image of dimensions 639x516
# img = Image.open(open('3wolfmoon.jpg'))
# dimensions are (height, width, channel)
# img = np.asarray(img, dtype='float32') / 256.

# put image in 4D tensor of shape (1, 3, height, width)
# img_ = img.transpose(2, 0, 1).reshape(1, 1, 3, 639, 516)
# img_ = img.transpose(2, 0, 1).reshape(batchsize, in_time, in_channels, in_height, in_width)
img = AD[0]
imgA_ = img.reshape(batchsize, in_time, in_channels, in_height, in_width)
imgB_ = img.reshape(batchsize, in_channels, in_time, in_height, in_width)
# filtered_imgA = fA(imgA_)
filtered_imgB = fB(imgB_)

im1 = filtered_img[:,:,0,:,:].transpose(0,2,3,1)
im2 = filtered_img[:,:,1,:,:].transpose(0,2,3,1)


# plot original image and first and second components of output
plt.subplot(1, 3, 1); plt.axis('off'); plt.imshow(img)
plt.gray()
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
plt.subplot(1, 3, 2); plt.axis('off'); plt.imshow(im1[0])
plt.subplot(1, 3, 3); plt.axis('off'); plt.imshow(im2[0])

# plt.subplot(1, 3, 2); plt.axis('off'); plt.imshow(filtered_img[0, 0, :, :])
# plt.subplot(1, 3, 3); plt.axis('off'); plt.imshow(filtered_img[0, 1, :, :])
plt.savefig('fig.png')
# plt.show()
