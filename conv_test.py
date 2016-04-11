__author__ = 'ehsanh'

import theano
from theano import tensor as T
from theano.tensor.nnet import conv, Conv3D, conv3d2d
import numpy
import pylab
from PIL import Image
import ipdb
# import theano.sandbox.cuda.fftconv.conv3d_fft as conv3d_fft
# T = theano.tensor
# floatX = theano.config.floatX
# import theano.tensor.nnet.conv3d2d

rng = numpy.random.RandomState(23455)

# define inputs and filters
batchsize     = 1
in_channels   = 1
in_time       = 3
in_width      = 516
in_height     = 639
flt_channels  = 2
flt_time      = 1
flt_width     = 9
flt_height    = 9

# instantiate 4D tensor for input
dtensor5 = T.TensorType('float32', (False,)*5)
# input = T.tensor4(name='input')
input = dtensor5(name='input')

# initialize shared variable for weights.
# w_shp = (2, 1, 9, 9)
# w_shp = (2, 1, 1, 9, 9)
w_shp = (flt_channels, flt_time, in_channels, flt_height, flt_width)
w_bound = numpy.sqrt(1 * 9 * 9)
W = theano.shared( numpy.asarray(
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
b = theano.shared(numpy.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=input.dtype), name ='b')

# build symbolic expression that computes the convolution of input with filters in w
# conv_out = conv.conv2d(input, W)
# conv_out = conv3d2d.conv3d(input, W)
# conv_out = Conv3D(input, W)
# conv_out = conv3d_fft(input, W)

output = T.nnet.conv3d2d.conv3d(
	signals=input,  # Ns, Ts, C, Hs, Ws
	filters=W, # Nf, Tf, C, Hf, Wf
	signals_shape=(batchsize, in_time, in_channels, in_height, in_width),
	filters_shape=(flt_channels, flt_time, in_channels, flt_height, flt_width),
	border_mode='valid')
output = output + b.dimshuffle('x','x',0,'x','x')

# output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x', 'x'))

# create theano function to compute filtered images
f = theano.function([input], output)


ipdb.set_trace()
# open random image of dimensions 639x516
img = Image.open(open('3wolfmoon.jpg'))
# dimensions are (height, width, channel)
img = numpy.asarray(img, dtype='float32') / 256.

# put image in 4D tensor of shape (1, 3, height, width)
# img_ = img.transpose(2, 0, 1).reshape(1, 1, 3, 639, 516)
img_ = img.transpose(2, 0, 1).reshape(batchsize, in_time, in_channels, in_height, in_width)
filtered_img = f(img_)

im1 = filtered_img[:,:,0,:,:].transpose(0,2,3,1)
im2 = filtered_img[:,:,1,:,:].transpose(0,2,3,1)


# plot original image and first and second components of output
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray()
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(im1[0])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(im2[0])

# pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
# pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.savefig('fig.png')
# pylab.show()
