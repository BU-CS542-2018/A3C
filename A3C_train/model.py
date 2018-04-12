#--------------------------------------------------------------------------------------------------------------------------------
# CS 542 Machine Learning Project, Winter 2018, Boston University
# Modified for the purpose of project
# Original code by OpenAI
# Description: Wrapper for Universe environment
#--------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')
#--------------------------------------------------------------------------------------------------------------------------------
# initializor for the get_variable function
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer
#--------------------------------------------------------------------------------------------------------------------------------
# flatten each tensor(if there are multiple tensors in x) in x into a vector
def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])]) # tf.reshape(x,[-1, size of each tensor])
#--------------------------------------------------------------------------------------------------------------------------------
# wrapper function for creating a convolutional layer
def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    # define the name of the filter and all properties within
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters] # filter shape 3 x 3 x 1 x 32

        # there are "num input feature maps * filter height * filter width" , which is 3 x 3 x 1 = 9 in this case
        # feature map # = 1 since we only feed in greyscale input
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size, 3 x 3 x 32 = 288
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))  # rougly 0.142
        
        #weight and bias
        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b
#--------------------------------------------------------------------------------------------------------------------------------
#
def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b  #matrix(tensor) multiplication
#--------------------------------------------------------------------------------------------------------------------------------
# d is 7 by default, 7 actions
def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)
#--------------------------------------------------------------------------------------------------------------------------------
# this is the main neural network architecture, using Long Short-term Memory
"""
Parameters:
With neon race game by default, the setup is the following
ob_space: 128 x 200
ac_space: 7
properties:
x:(not to be confused with x which is the conv layers) placeholder for input(images)
state_size: lstm state size
state_init: initial state of LSTM, generated with all 0s.
state_in: placeholder version of state_init
logits: probability of each action
vf: state value
state_out
sample:
var_list:
"""
class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space):
        
        # x is 128 x 200 x 1
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        
        #4 convolutional layers with elu activation function
        # Elu: https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/contrib/keras/layers/ELU
        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(flatten(x), [0])
        
        # create LSTM unit
        # Paper: https://arxiv.org/pdf/1409.2329.pdf
        size = 256
        if use_tf100_api:
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        #state initialization
        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        #setup placeholder for further update
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        if use_tf100_api:
            state_in = rnn.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
            
        #process through RNN
        #note: lstm_outputs is a tensor with size batch_size x max_time x 256
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        
        # reshape to 2D
        x = tf.reshape(lstm_outputs, [-1, size])
        
        # convert to a tensor, which will further be computed using softmax in a3c object.
        self.logits = .(x, ac_space, "action", normalized_columns_initializer(0.01))
        # this will return one value as our state value
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])   
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        # this will store all the name of trainable variables
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    #accessors
    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]
