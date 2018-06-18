import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import sys


def next_batch(training_data, batch_size, steps):

    # si
    # a, b = next_batch(np.linspace(0, 10, 11), 2, 4)

    # training_data
    # [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]

    # Grab a random starting point for each batch
    rand_start = np.random.randint(0, len(training_data) - steps)

    # Create Y data for time series in the batches
    # y_batch
    # [[0. 1. 2. 3. 4.]]
    y_batch = np.array(training_data[rand_start:rand_start + steps + 1]).reshape(1, steps + 1)

    # esto regresa
    # [[[0.]
    #   [1.]
    #   [2.]
    #   [3.]]]
    # [[[1.]
    #   [2.]
    #   [3.]
    #   [4.]]]

    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)

# leer datos
milk = pd.read_csv('csv/monthly-milk-production-pounds-p.csv', index_col='Month', header=0)
print(milk.head())

# ajustar index como tipo de dato datetime
milk.index = pd.to_datetime(milk.index)
milk.plot()
plt.show()

print(milk.info())

# separar para entrenar
# los primeros en orden son para entrenar y los ultimos de prueba
train_set = milk.head(156)
test_set = milk.tail(12)

# normalizar
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.transform(test_set)


# hiper parametros ---------------------

# Just one feature, the time series
num_inputs = 1
# Num of steps in each batch
num_time_steps = 12
# 100 neuron layer, play with this
num_neurons = 100
# Just one output, predicted time series
num_outputs = 1

# You can also try increasing iterations, but decreasing learning rate
# learning rate you can play with this
learning_rate = 0.03
# how many iterations to go through (training steps), you can play with this
num_train_iterations = 4000
# Size of the batch of data
batch_size = 1
# --------------------------------------

X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

# Also play around with GRUCell
cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y))  # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

# entrenar
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#     sess.run(init)
#
#     for iteration in range(num_train_iterations):
#
#         X_batch, y_batch = next_batch(train_scaled, batch_size, num_time_steps)
#         sess.run(train, feed_dict={X: X_batch, y: y_batch})
#
#         if iteration % 100 == 0:
#             mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
#             print(iteration, "\tMSE:", mse)
#
#     # Save Model for Later
#     saver.save(sess, "./ex_time_series_model")

# predecir
with tf.Session() as sess:
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "./ex_time_series_model")

    # Create a numpy array for your genreative seed from the last 12 months of the
    # training set data. Hint: Just use tail(12) and then pass it to an np.array
    train_seed = list(train_scaled[-12:])

    # print(np.array(train_scaled))
    # print(np.array(train_seed))
    # sys.exit(0)

    # Toma los ultimos elementos y comienza a iterar desde el primero para agregar un elemento
    # predecido al final
    for iteration in range(12):
        print iteration
        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        train_seed.append(y_pred[0, -1, 0])
        print np.array(train_seed)

# Grab the portion of the results that are the generated values and apply inverse_transform on them to turn
# them back into milk production value units (lbs per cow). Also reshape the results to be (12,1) so we can easily
# add them to the test_set dataframe.
results = scaler.inverse_transform(np.array(train_seed[12:]).reshape(12,1))

test_set['Generated'] = results

test_set.plot()
plt.show()
