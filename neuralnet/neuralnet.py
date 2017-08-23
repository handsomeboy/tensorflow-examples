from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# define placeholders for our training variables
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
    """Initializes weights randomly from a normal distribution
    Params: shape: list of dimensionality of tensor
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Initializes the bias term randomly from a normal distribution.
    Params: shape: list of dimensionality for the bias term.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def fc_layer(scope, x, weight_shape, activation = 'relu', keep_prob = 1.0):
    # TODO activation param isn't supported...
    with tf.variable_scope(scope):
        W_fc = weight_variable(weight_shape)
        b_shape = [weight_shape[-1]]
        b_fc = bias_variable(b_shape)
        h_fc = tf.nn.sigmoid(tf.matmul(x, W_fc) + b_fc)
        h_fc_drop = tf.nn.dropout(h_fc, keep_prob=keep_prob)
        return h_fc_drop

# create weights and biases and function for our first layer
W_fc1, b_fc1 = weight_variable([784, 100]), bias_variable([100])
# hidden layer computes relu(Wx + b)
#h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

keep_prob_1 = tf.placeholder(tf.float32)
# add dropout: discard activations with probability given by keep_prob
#h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob_1)

h_fc1_dropout = fc_layer("layer-1", x, [784, 100], activation = 'softplus',
                         keep_prob = keep_prob_1)

# create w, b, and function for our next layer
W_fc2, b_fc2 = weight_variable([100, 30]), bias_variable([30])
#h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)

# # add dropout
keep_prob_2 = tf.placeholder(tf.float32)

# # discard second hidden layer activations with keep_prob_2 probability
#h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob_2)
h_fc2_dropout = fc_layer("layer-2", h_fc1_dropout, [100, 30], activation = 'softplus',
                         keep_prob = keep_prob_2)
# define w and b for the softmax layer
W_fc3, b_fc3 = weight_variable([30, 10]), bias_variable([10])

# softmax Output
y_pred = tf.matmul(h_fc2_dropout, W_fc3) + b_fc3
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = y_)
mse = tf.reduce_sum(tf.square(y_ - y_pred))
loss_fun = cross_entropy # change loss function here, keep it in loss_fun var
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_fun)
#train_step = tf.train.MomentumOptimizer(1e-4, 0.5, name='Momentum', use_nesterov=True).minimize(loss_fun)
# accuracy variables
cp = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(cp, tf.float32))
init = tf.global_variables_initializer()


with tf.Session() as sess:
    tf.set_random_seed(1)
    sess.run(init)
    losses = []
    for i in range(10000):
        # iterate for 10k epochs and run batch SGD.
        batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob_1: 0.8,
                                        keep_prob_2: 0.5})
        if i % 100 == 0:
            print("epoch: {}".format(i + 1))
            print(acc.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob_1: 1.0,
                                      keep_prob_2: 1.0}))
            cur_loss = loss_fun.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob_1: 1.0,
                                      keep_prob_2: 1.0})
            # hack because cur_loss is a list when we use tf.nn.cross_entropy but a single numpy float32 when we use mse.
            try:
                loss = sum(cur_loss)
            except TypeError:
                loss = cur_loss

            print(loss)
            losses.append(loss)
    print("done training!")
    test_acc = acc.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob_1: 1.0,
                                   keep_prob_2: 1.0})
    print("test acc: {}".format(test_acc))
    print("losses")
    print(losses)
    print("using {} loss".format("mse" if loss_fun == mse else "cross_entropy"))
    x_loss, y_loss = [i for i in range(len(losses))], losses
    slopes = [losses[i + 1] - losses[i] for i in range(len(losses) - 1)]
    print("slopes")
    print(slopes)
    plt.plot(x_loss, y_loss)
    plt.plot([i for i in range(len(losses) - 1)], slopes)
    plt.show()

sess.close()
