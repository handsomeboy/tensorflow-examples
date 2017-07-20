import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

LOGS_PATH = '/tmp/inceptionlog'

# define input variables
x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 10],name='y-labels')

def weight_variable(shape, name = 'Weights'):
    """Initializes weights randomly from a normal distribution
    Params: shape: list of dimensionality of the tensor to be initialized
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)


def bias_variable(shape, name = 'bias var'):
    """Initializes the bias term randomly from a normal distribution.
    Params: shape: list of dimensionality for the bias term.
    """
    initial = tf.constant(0.1, shape=shape, name = 'Bias')
    return tf.Variable(initial, name = name)

def conv2d(x, W):
    """Performs a convolution over a given patch x with some filter W.
    Uses a stride of length 1 and SAME padding (padded with zeros at the edges)
    Params:
    x: tensor: the image to be convolved over
    W: the kernel (tensor) with which to convolve.
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name = 'max-pool-2x2'):
    """Performs a max pooling operation over a 2 x 2 region"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name = name)

def max_pool_3x3(x, name = 'max-pool-3x3'):
    """Performs a max pooling operation over a 3x3 region"""
    return tf.nn.max_pool(x, ksize = [1, 3, 3, 1],
                          strides = [1, 1, 1, 1], padding = 'SAME', name = name)

x_image = tf.reshape(x, [-1,28,28,1], name = 'x-image-reshaped') # convert x to a 4-d tensor

with tf.name_scope('model'):
    # Inception module 1
    # 1x1 convolution -> 32 feature maps
    W_conv1_1x1_1 = weight_variable([1, 1, 1, 32], name = 'W_conv1_1x1_1')
    b_conv1_1x1_1 = bias_variable([32], name = 'b_conv1_1x1_1')
    # 1x1 convolution -> 16 feature maps, 3 x 3 convolution -> 32 feature maps
    W_conv1_1x1_2 = weight_variable([1, 1, 1, 16], name = 'W_conv1_1x1_2')
    b_conv1_1x1_2 = bias_variable([16], name = 'b_conv1_1x1_2')
    W_conv1_3x3 = weight_variable([3, 3, 16, 32], name = 'W_conv1_3x3')
    b_conv1_3x3 = bias_variable([32], name = 'b_conv1_3x3')
    # 1x1 convolution -> 16 feature maps, 5 x 5 convolution -> 32 feature maps
    W_conv1_1x1_3 = weight_variable([1, 1, 1, 16], name = 'W_conv1_1x1_3')
    b_conv1_1x1_3 = bias_variable([16], name = 'b_conv1_1x1_3')
    W_conv1_5x5 = weight_variable([5, 5, 16, 32], name = 'W_conv1_1x1_5x5')
    b_conv1_5x5 = weight_variable([32], name = 'b_conv1_1x1_5x5')
    # max-pooling (of the input x)-> 1x1 convolution -> 32 feature maps
    W_conv1_1x1_4 = weight_variable([1, 1, 1, 32], name = 'W_conv1_1x1_4')
    b_conv1_1x1_4 = bias_variable([32], name = 'b_conv1_1x1_4')
    with tf.name_scope('inception-1'):
        # compute 1x1, 1x1 -> 3x3, 1x1 -> 5x5, and max-pool of x followed by 1x1
        # direct 1 x 1 conv
        h_1x1_1 = conv2d(x_image, W_conv1_1x1_1) + b_conv1_1x1_1
        # 1x1 -> 3 x 3, resulting in 32 feature maps
        h_1x1_2 = conv2d(x_image, W_conv1_1x1_2) + b_conv1_1x1_2
        h_3x3_1 = conv2d(h_1x1_2, W_conv1_3x3) + b_conv1_3x3
        # 1x1 -> 5 x 5, resulting in 32 feature maps
        h_1x1_3 = conv2d(x_image, W_conv1_1x1_3) + b_conv1_1x1_3
        h_5x5_1 = conv2d(h_1x1_3, W_conv1_5x5) + b_conv1_5x5
        # max pooling -> 1 x 1 conv resulting in 32 feature maps
        x_image_maxpool = max_pool_3x3(x_image)
        h_maxpool = conv2d(x_image_maxpool, W_conv1_1x1_4) + b_conv1_1x1_4
        inception_1 = tf.nn.relu(tf.concat([h_1x1_1, h_3x3_1, h_5x5_1, h_maxpool], 3))
