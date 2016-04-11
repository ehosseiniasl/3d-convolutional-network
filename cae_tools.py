"""Useful tools for main program

Author : Hu Yuhuang
Date   : 2014-10-21

Notes
-----

Most of the code in this file are modified from LISA lab's
Deep Learning Tutorial.

"""

# System library
import cPickle, gzip, os;

# public library
import numpy;

import theano;
import theano.tensor as T;
from theano.compat.python2x import OrderedDict

# private library
import conv_aes.nntool as nnt;

class HiddenLayer(object):
    def __init__(self,
                 rng,
                 data_in,
                 n_in,
                 n_out,
                 W=None,
                 b=None,
                 activate_mode='tanh'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activate_mode: string of activation mode
        :param activate_mode: supports 4 non-linearity function: tanh, sigmoid, relu, softplus
        """
        self.input = data_in;
        self.activate_mode=activate_mode;

        if (self.activate_mode=="tanh"):
            self.activation=nnt.tanh;
            self.d_activation=nnt.d_tanh;
        elif (self.activate_mode=="relu"):
            self.activation=nnt.relu;
        elif (self.activate_mode=="sigmoid"):
            self.activation=nnt.sigmoid;
            self.d_activation=nnt.d_sigmoid;
        elif (self.activate_mode=="softplus"):
            self.activation=nnt.softplus;
            self.d_activation=nnt.d_softplus;
        elif (self.activate_mode=="softmax"):
            self.activation=nnt.softmax;
        else:
            raise ValueError("Value %s is not a valid choice of activation function"
                             % self.activate_mode);

        if W is None:
            W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)),
                                                 high=numpy.sqrt(6. / (n_in + n_out)),
                                                 size=(n_in, n_out)),
                                     dtype='float32');
            if (self.activate_mode=="sigmoid"):
                W_values *= 4;

            W = theano.shared(value=W_values, name='W', borrow=True);

        if b is None:
            b_values = numpy.zeros((n_out,), dtype='float32');
            b = theano.shared(value=b_values, name='b', borrow=True);

        self.W = W;
        self.b = b;

        self.lin_output=self.getLinOut(self.input);
        self.output=self.getActivation(self.lin_output);

        self.params = [self.W, self.b];

    def getLinOutput(self,
                     data_in,
                     weights,
                     bias):
        """Get total weighted sum of input

        Parameters
        ----------
        data_in : theano tensor of input
            input data
        weights : theano tensor
            weights associated with this layer
        bias : theano tensor
            bias associated with this layer
        """

        return T.dot(data_in, weights)+bias;

    def getLinOut(self,
                  data_in):
        return self.getLinOutput(data_in=data_in,
                                 weights=self.W,
                                 bias=self.b);

    def getActivation(self,
                      data_in):
        return self.activation(data_in);

class HiddenRWLayer(HiddenLayer):
    """A typical MLP layer added feedforward alignment support
    """

    def __init__(self,
                 rng,
                 data_in,
                 n_in,
                 n_out,
                 W=None,
                 b=None,
                 B_limits=0.1,
                 activate_mode='tanh'):
        """Follow HiddenLayer's definition
        """

        super(HiddenRWLayer, self).__init__(rng=rng,
                                          data_in=data_in,
                                          n_in=n_in,
                                          n_out=n_out,
                                          W=W,
                                          b=b,
                                          activate_mode=activate_mode);

        B_values = numpy.asarray(rng.uniform(low=-1*B_limits,#-numpy.sqrt(6. / (n_in + n_out)),
                                             high=B_limits,#numpy.sqrt(6. / (n_in + n_out)),
                                             size=(n_out, n_in)),
                                 dtype='float32');
        #if (self.activate_mode=="sigmoid"):
        #    B_values *= 4;
        self.B = theano.shared(value=B_values, name='B', borrow=True);

        self.params=[self.W, self.b];


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, data_in, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype='float32'
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype='float32'
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(data_in, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1
    
        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred'
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class MLP(object):
    """MLP with an hidden layer
    """

    def __init__(self,
                 rng,
                 data_in,
                 n_in,
                 n_hidden,
                 n_out,
                 W_hidden=None,
                 b_hidden=None,
                 W_out=None,
                 b_out=None):
        """Initialization of MLP network
        """

        self.input=data_in;
        self.n_in=n_in;
        self.n_hidden=n_hidden;
        self.n_out=n_out;

        self.hidden_layer=HiddenLayer(rng,
                                      data_in=self.input,
                                      n_in=self.n_in,
                                      n_out=self.n_hidden,
                                      W=W_hidden,
                                      b=b_hidden,
                                      activate_mode='tanh');

        self.out_layer=HiddenLayer(rng,
                                   data_in=self.hidden_layer.output,
                                   n_in=n_hidden,
                                   n_out=n_out,
                                   W=W_out,
                                   b=b_out,
                                   activate_mode='softmax');

        #self.logReg_layer=LogisticRegression(data_in=self.hidden_layer.output,
        #                                     n_in=self.n_hidden,
        #                                     n_out=self.n_out);

        #self.L1=(abs(self.hidden_layer.W).sum()+abs(self.logReg_layer.W).sum());
        #self.L2=((self.hidden_layer.W**2).sum()+(self.logReg_layer.W**2).sum());

        self.L1=(abs(self.hidden_layer.W).sum()+abs(self.out_layer.W).sum());
        self.L2=((self.hidden_layer.W**2).sum()+(self.out_layer.W**2).sum());

        self.pred=self.out_layer.output;
        self.prediction=T.argmax(self.pred, axis=1);

        #self.negative_log_likelihood=self.logReg_layer.negative_log_likelihood;
        #self.errors=self.logReg_layer.errors;

        self.params=self.hidden_layer.params+self.out_layer.params;

    def errors(self,
               target):
        """Get errors
        """

        return T.mean(T.neq(self.prediction, target));

    def get_cost(self,
                 target):
        """Softmax cross entropy
        """

        return -T.mean(T.log(self.pred)[T.arange(target.shape[0]), target]);

    def get_cost_update(self,
                        target,
                        L1_reg=0.00,
                        L2_reg=0.001,
                        learning_rate=0.1):
        """Get cost and update
        """

        #cost=self.negative_log_likelihood(target)+L1_reg*self.L1+L2_reg*self.L2;
        cost = self.get_cost(target)+L1_reg*self.L1+L2_reg*self.L2;

        gparams=T.grad(cost, self.params);

        updates=[(param_i, param_i-learning_rate*grad_i)
                 for param_i, grad_i in zip(self.params, gparams)];

        return (cost, updates);
        
class RWMLP(object):
    """MLP layer with feedfoward alignment
    """

    def __init__(self,
                 rng,
                 data_in,
                 n_in,
                 n_hidden,
                 n_out,
                 W_hidden=None,
                 b_hidden=None,
                 B_hidden_limits=0.1,
                 W_out=None,
                 b_out=None,
                 B_out_limits=0.1):
        """Initialization of MLP network
        """

        self.input=data_in;
        self.n_in=n_in;
        self.n_hidden=n_hidden;
        self.n_out=n_out;

        self.hidden_layer=HiddenRWLayer(rng,
                                        data_in=data_in,
                                        n_in=n_in,
                                        n_out=n_hidden,
                                        W=W_hidden,
                                        b=b_hidden,
                                        B_limits=B_hidden_limits,
                                        activate_mode='tanh');

        self.out_layer=HiddenRWLayer(rng,
                                     data_in=self.hidden_layer.output,
                                     n_in=n_hidden,
                                     n_out=n_out,
                                     W=W_out,
                                     b=b_out,
                                     B_limits=B_out_limits,
                                     activate_mode='softmax');

        self.L1=(abs(self.hidden_layer.W).sum()+abs(self.out_layer.W).sum());
        self.L2=((self.hidden_layer.W**2).sum()+(self.out_layer.W**2).sum());

        self.pred=self.out_layer.output;
        self.prediction=T.argmax(self.pred, axis=1);

        self.params=self.hidden_layer.params+self.out_layer.params;

    def errors(self,
               target):
        """Get errors
        """

        return T.mean(T.neq(self.prediction, target));

    def get_cost(self,
                 target):
        """Softmax cross entropy
        """

        return -T.mean(T.log(self.pred)[T.arange(target.shape[0]), target]);

    def get_cost_update(self,
                        target,
                        L1_reg=0.00,
                        L2_reg=0.0001,
                        learning_rate=0.1):

        cost = self.get_cost(target)+L1_reg*self.L1+L2_reg*self.L2;

        d_target_net=T.grad(cost, self.out_layer.lin_output);

        d_b_2=d_target_net.sum(axis=0);
        d_W_2=T.dot(self.hidden_layer.output.T, d_target_net);

        d_H_1=T.dot(d_target_net, self.out_layer.B);
        d_H1_net=d_H_1*self.hidden_layer.d_activation(self.hidden_layer.lin_output);

        d_b_1=d_H1_net.sum(axis=0);
        d_W_1=T.dot(self.input.T, d_H1_net);

        gparams=[d_W_1, d_b_1, d_W_2, d_b_2];

        updates=[(param_i, param_i-learning_rate*grad_i)
                 for param_i, grad_i in zip(self.params, gparams)];

        return (cost, updates);



def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               #dtype='float32'
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               #dtype='float32'
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
