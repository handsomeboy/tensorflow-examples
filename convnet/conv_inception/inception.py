import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

LOGS_PATH = '/tmp/inceptionlog'

# define input variables
x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 10],name='y-labels')
keep_prob = tf.placeholder(tf.float32) # 1 - dropout probability
# shape [m, m, x, y] - generally m * m convolution from x feature maps to y feature maps.
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
    print("going through inception module 1")
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
        print("going through inception module 1")
        # compute 1x1, 1x1 -> 3x3, 1x1 -> 5x5, and max-pool of x followed by 1x1
        # direct 1 x 1 conv
        h_1x1_1 = conv2d(x_image, W_conv1_1x1_1) + b_conv1_1x1_1
        # 1x1 -> relu -> 3 x 3, resulting in 32 feature maps
        h_1x1_2 = tf.nn.relu(conv2d(x_image, W_conv1_1x1_2) + b_conv1_1x1_2)
        h_3x3_1 = conv2d(h_1x1_2, W_conv1_3x3) + b_conv1_3x3
        # 1x1 -> relu -> 5 x 5, resulting in 32 feature maps
        h_1x1_3 = tf.nn.relu(conv2d(x_image, W_conv1_1x1_3) + b_conv1_1x1_3)
        h_5x5_1 = conv2d(h_1x1_3, W_conv1_5x5) + b_conv1_5x5
        # max pooling -> 1 x 1 conv resulting in 32 feature maps
        x_image_maxpool = max_pool_3x3(x_image)
        h_maxpool = conv2d(x_image_maxpool, W_conv1_1x1_4) + b_conv1_1x1_4
        # concat along 3rd dim -> relu -> final output of inception1
        inception_1 = tf.nn.relu(tf.concat([h_1x1_1, h_3x3_1, h_5x5_1, h_maxpool], 3))

    # 1x1 conv on inception 1 which has 4 * 32 feature maps (due to the concat) to 16 feature maps
    W_conv2_1x1_1 = weight_variable([1, 1, 4 * 32, 16], name = 'W_conv2_1x1_1')
    b_conv2_1x1_1 = bias_variable([16], name = 'b_conv2_1x1_1')
    # 1x1 conv on inception 1 -> 16 feature maps, followed by 3x3 conv to 64 feature maps.
    W_conv2_1x1_2 = weight_variable([1, 1, 4 * 32, 16], name = 'W_conv2_1x1_2')
    b_conv2_1x1_2 = weight_variable([16], name = 'b_conv2_1x1_2')
    W_conv2_3x3 = weight_variable([3, 3, 16, 64], name = 'W_conv2_3x3')
    b_conv2_3x3 = bias_variable([64], name = 'b_conv2_3x3')
    # 1x1 conv on inception 1 -> 16 feature maps, followed by 5x5 conv to 64 feature maps.
    W_conv2_1x1_3 = weight_variable([1, 1, 4 * 32, 16], name = 'W_conv2_1x1_3')
    b_conv2_1x1_3 = bias_variable([16], name = 'b_conv2_1x1_3')
    W_conv2_5x5 = weight_variable([5, 5, 16, 64], name = 'W_conv2_5x5')
    b_conv2_5x5 = bias_variable([64], name = 'b_conv2_5x5')
    # max pooling -> 1x1 conv resulting in 64 feature maps
    W_conv2_1x1_4 = weight_variable([1, 1, 4 * 32, 64], name = 'W_conv2_1x1_4')
    b_conv2_1x4_4 = bias_variable([64], name = 'b_conv2_1x1_4')
    with tf.name_scope('inception-2'):
        print("going through inception module 2")
        # compute direct 1x1 conv
        h_1x1_1_2 = conv2d(inception_1, W_conv2_1x1_1) + b_conv2_1x1_1
        # compute 1x1 conv -> relu -> 3x3 conv
        h_1x1_2_2 = tf.nn.relu(conv2d(inception_1, W_conv2_1x1_2) + b_conv2_1x1_2)
        h_3x3_2 = conv2d(h_1x1_2_2, W_conv2_3x3) + b_conv2_3x3
        # compute 1x1 conv -> relu -> 5x5 conv
        h_1x1_1_3 = tf.nn.relu(conv2d(inception_1, W_conv2_1x1_3) + b_conv2_1x1_3)
        h_5x5_2 = conv2d(h_1x1_1_3, W_conv2_5x5) + b_conv2_5x5
        pooled = max_pool_3x3(inception_1)
        inception_2 = tf.nn.relu(tf.concat([h_1x1_1_2, h_3x3_2, h_5x5_2, pooled], 3))

    flattened = tf.reshape(inception_2, [-1, 28 * 28 * 4 * 5984])
    # first fully-connected layer
    print("going through first FC layer")
    W_fc1 = weight_variable([28 * 28 * 4 * 5984, 200], 'W_fc1')
    b_fc1 = bias_variable([200], 'b_fc1')
    with tf.name_scope('fc-1'):
        h_fc1 = tf.nn.relu(tf.matmul(flattened, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob = keep_prob)

    # final fc layer -> softmax
    print("going through final FC layer")
    W_fc2 = weight_variable([200, 10], 'W_fc2')
    b_fc2 = bias_variable([10], 'b_fc2')
    with tf.name_scope('softmax-2'):
        h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

with tf.name_scope('loss'):
    print("the loss")
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits =h_fc2, labels = y_)

with tf.name_scope('optimizer'):
    print("the optimizer")
    lr = 1e-4
    opt = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

with tf.name_scope('acc'):
    print("the accuracy")
    cp = tf.equal(tf.argmax(h_fc2, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(cp, tf.float32))

init = tf.global_variables_initializer()
# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cross_entropy)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

n_epochs = 20000
minibatch_size = 88
print("about to launch the session")
with tf.Session() as sess:
    print("about to run sess.run init")
    sess.run(init)
    print("ran sess.run init")
    # grab the file writer
    summary_writer = tf.summary.FileWriter('/tmp/inceptionlog', graph=tf.get_default_graph())
    print("HERE ABOUT TO START")
    exit()
    for i in range(n_epochs):
        batch = mnist.train.next_batch(minibatch_size)
        if i % 100 == 0:
            print("epoch: {}".format(i))
            train_acc = accuracy.eval(feed_dict = { x: batch[0], y_: batch[1], keep_prob : 1.0 })
            print("train acc: {}".format(train_acc))
        _, c, summary = sess.run([opt, cross_entropy, merged_summary_op], feed_dict = {
            x: batch[0],
            y_: batch[1],
            keep_prob: 0.5,
        })
        summary_writer.add_summary(summary, i)
    test_acc = accuracy.eval(feed_dict = { x: batch[0], y_: batch[1], keep_prob: 1.0})
    print("test acc: {}".format(test_acc))
    print("run tensorboard --logdir=/tmp/inceptionlog and then go to localhost:6006 ")
    sess.close()
