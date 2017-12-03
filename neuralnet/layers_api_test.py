import tensorflow as tf
import numpy as np
n_epochs = 20000
minibatch_size = 50
lr = 1e-4
dropout_prob = 0.5

x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 10],name='y-labels')
x_image = tf.reshape(x, [-1,28,28,1], name = 'x-image-reshaped') # covert x to a 4-d tensor

with tf.name_scope('model'):
	fc_layer_1 = tf.layers.dense(inputs = x, units = 100, activation = tf.nn.relu)
	fc_drop = tf.layers.dropout(inputs = fc_layer_1, rate = 0.4)
	W_out = tf.Variable(tf.truncated_normal(shape = [100, 10], stddev = 0.1))
	b_out = tf.Variable(tf.constant(0.1, shape = [10]))
	y_pred = tf.matmul(fc_drop, W_out) + b_out
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = y_))
	opt = tf.train.AdamOptimizer(1e-4).minimize(loss)
	correct_prediction = tf.equal(tf.argmax(y_pred, axis = 1), tf.argmax(y_, axis = 1))
	acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	init = tf.global_variables_initializer()
	sess = tf.InteractiveSession()
	sess.run(init)
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	for i in range(10000):
		batch = mnist.train.next_batch(50)
		sess.run(opt, feed_dict = {x: batch[0], y_: batch[1]})
		if i % 50 == 0:
			current_loss = loss.eval(feed_dict = {x: batch[0], y_: batch[1]})
			print("accuracy: {}".format(acc.eval(feed_dict = {x: batch[0], y_: batch[1]})))


