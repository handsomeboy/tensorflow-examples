import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gen_data import gen_data

RNNconfig = {
    'num_steps' : 5, # higher n = capture longer term dependencies, but more expensive (and potential vanishing gradient issues)
    'batch_size' : 200, # amount of data to backprop on.
    'state_size' :10, # number of hidden states.
    'learning_rate' : 0.1
}

num_classes = 2

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    """Generates batches from raw data"""
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size # amount of data in each batch.
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(n, num_steps):
    # prefer yield to save memory.
    for i in range(n):
        yield gen_batch(gen_data(), RNNconfig['batch_size'], num_steps)

# placeholder
x = tf.placeholder(tf.int32, [None, RNNconfig['num_steps']], name='input_placeholder')
y = tf.placeholder(tf.int32, [None, RNNconfig['num_steps']], name='labels_placeholder')
# need to initialize an initial state, which can just be all zeros.
init_state = tf.zeros([RNNconfig['batch_size'], RNNconfig['state_size']])

# Turn our x placeholder into a list of one-hot
# rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]
# rnn_inputs is a list of training data.
x_one_hot = tf.one_hot(x, num_classes) # note: num_classes is not an RNN variable...
rnn_inputs = tf.unstack(x_one_hot, axis=1)

cell = tf.contrib.rnn.BasicRNNCell(RNNconfig['state_size'])
# rnn_outputs is a list of hidden states.
rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state = init_state)


#logits and predictions
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [RNNconfig['state_size'], num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs] # logits are a list of predictions (one for each state)
predictions = [tf.nn.softmax(logit) for logit in logits]

# Turn our y placeholder into a list of labels
y_as_list = tf.unstack(y, num=RNNconfig['num_steps'], axis=1)

#losses and train_step
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
          logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.1).minimize(total_loss)
init = tf.global_variables_initializer()

def train_network(num_epochs, verbose=True):

    with tf.Session() as sess:
        sess.run(init)
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, RNNconfig['num_steps'])):
            training_loss = 0
            training_state = np.zeros((RNNconfig['batch_size'], RNNconfig['state_size']))
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses,
                              total_loss,
                              final_state,
                              train_step],
                                  feed_dict={x:X, y:Y, init_state:training_state})
                preds = sess.run(predictions, feed_dict = {x: X})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last 100 steps:", training_loss/100)
                    training_losses.append(training_loss/100)
                    training_loss = 0

        # TODO rewrite the following code. It is very unclear and has overly complicated list comprehensions which make no sense.
        preds = [list(k) for k in preds]
        flat_preds = [round(val) for li in preds for nparray in li for val in nparray]
        flat_Y = [item for sublist in Y for item in sublist]
        preds, Y = flat_preds, flat_Y
        print("predicted: {}".format(preds[:25]))
        print("actual: {}".format(Y[:25]))
        # compute squared error
        preds = preds[:len(Y)]
        Y, preds = np.array(Y), np.array(preds)
        assert len(Y) == len(preds), "Y is {} but preds is {}".format(len(Y), len(preds))
        print("err: {}".format(sum(abs(preds - Y))))


    return training_losses
training_losses = train_network(num_epochs = 5)
