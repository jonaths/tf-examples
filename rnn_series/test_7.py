# Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import numpy as np
import random
from random import shuffle
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import rnn

# NUM_EXAMPLES = 10000
#
# # todas las posibles combinaciones de longitud 20 de 0 y 1
# train_input = ['{0:020b}'.format(i) for i in range(2 ** 20)]
# shuffle(train_input)
# train_input = [map(int, i) for i in train_input]
# ti = []
# for i in train_input:
#     temp_list = []
#     for j in i:
#         temp_list.append([j])
#     ti.append(np.array(temp_list))
# train_input = ti
#
# # se plantea como un problema de clasificacion
# # one hot encoding para cada clase: la clase 1000... corresponde a una
# # cuenta de cero
# train_output = []
# for i in train_input:
#     count = 0
#     for j in i:
#         if j[0] == 1:
#             count += 1
#     temp_list = ([0] * 21)
#     temp_list[count] = 1
#     train_output.append(temp_list)
#
# # in our train_input and train_output, we have 2^20  (1,048,576) unique examples
# # tomamos los primeros 10,000 para entrenar
# # el resto para probar
# test_input = train_input[NUM_EXAMPLES:]
# test_output = train_output[NUM_EXAMPLES:]
# train_input = train_input[:NUM_EXAMPLES]
# train_output = train_output[:NUM_EXAMPLES]
# print(len(test_input), len(train_input))
#
# print "test and training data loaded ============================================"
#
# batches, input, dimension of each input
data = tf.placeholder(tf.float32, [None, 20, 1])

# batches, numero de clases
target = tf.placeholder(tf.float32, [None, 21])

num_hidden = 24
cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

# unroll the network and pass the data to it and store the output in val.
# We also get the state at the end of the dynamic run as a return value but we discard it
# because every time we look at a new sequence, the state becomes irrelevant
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

# tranpose (input, batches, dimension of each input)
val = tf.transpose(val, [1, 0, 2])

# we take the values of outputs only at sequences last input, which means in a
# string of 20 were only interested in the output we got at the 20th character and the
# rest of the output for previous characters is irrelevant here.
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

# with tf.Session() as sess:
#
#     sess.run(init_op)
#
#     batch_size = 1000
#     no_of_batches = int(len(train_input)) / batch_size
#     epoch = 5000
#     for i in range(epoch):
#         ptr = 0
#         for j in range(no_of_batches):
#             inp, out = train_input[ptr:ptr + batch_size], train_output[ptr:ptr + batch_size]
#             ptr += batch_size
#             sess.run(minimize, {data: inp, target: out})
#         print "Epoch ", str(i)
#         # Save Model for Later
#         if epoch % 1000 == 0:
#             saver.save(sess, "./ex_time_series_model")

with tf.Session() as sess:
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "./ex_time_series_model")

    # incorrect = sess.run(error, {data: test_input, target: test_output})
    pred = sess.run(prediction, {
        data: [[[0], [0], [0], [1], [1], [0], [1], [1], [1], [0], [1], [0], [0], [1], [1], [0], [1], [1], [1], [0]]]})
    # print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
    plt.bar(range(0, 21), pred.flatten())
    plt.show()
    # print (pred)
    # print(pred.flatten())
