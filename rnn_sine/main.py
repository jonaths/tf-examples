import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys


class TimeSeriesData():
    def __init__(self, num_points, xmin, xmax):

        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = 1.0 * (xmax - xmin) / num_points
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sin(x_series)

    def next_batch(self, batch_size, steps, return_batch_ts=False):

        # Grab a random starting point for each batch
        rand_start = np.random.rand(batch_size, 1)

        # Convert to be on time series
        ts_start = rand_start * (self.xmax - self.xmin - (steps * self.resolution))
        # print(ts_start)

        # Create batch Time Series on t axis
        batch_ts = 1.0 * ts_start + np.arange(0.0, steps + 1) * self.resolution

        # Create Y data for time series in the batches
        y_batch = np.sin(batch_ts)

        # Format for RNN
        y1 = y_batch[:, :-1].reshape(-1, steps, 1)
        y2 = y_batch[:, 1:].reshape(-1, steps, 1)
        # print(np.array(y1))
        # print(np.array(y2))
        if return_batch_ts:
            return y1, y2, batch_ts
        else:
            return y1, y2


ts_data = TimeSeriesData(250, 0, 10)
# plt.plot(ts_data.x_data, ts_data.y_true)
# plt.show()

# Num of steps in batch (also used for prediction steps into the future)
num_time_steps = 30
y1, y2, ts = ts_data.next_batch(1, num_time_steps, True)
plt.plot(ts_data.x_data, ts_data.y_true, label='Sin(t)')
plt.plot(ts.flatten()[1:], y2.flatten(), '*', label='Single Training Instance')
plt.legend()
plt.tight_layout()
plt.show()

# We are trying to predict a time series shifted over by t+1
train_inst = np.linspace(5, 5 + ts_data.resolution * (num_time_steps + 1), num_time_steps+1)
plt.title("A training instance", fontsize=14)
plt.plot(train_inst[:-1], ts_data.ret_true(train_inst[:-1]), "bo", markersize=15,alpha=0.5 ,label="instance")
plt.plot(train_inst[1:], ts_data.ret_true(train_inst[1:]), "ko", markersize=7, label="target")
plt.show()

# creating model
# dados los puntos azules tiene que generar los negros
tf.reset_default_graph()

# hiperparametros

# Just one feature, the time series
num_inputs = 1
# 100 neuron layer, play with this
num_neurons = 100
# Just one output, predicted time series
num_outputs = 1
# learning rate, 0.0001 default, but you can play with this
learning_rate = 0.0001
# how many iterations to go through (training steps), you can play with this
num_train_iterations = 2000
# Size of the batch of data
batch_size = 1

# batch, steps, inputs
# [[[i0_0, i0_1],
#   [i1_0, i1_1]],
# ]

X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

# RNN Cell Layer

# out1, out2 -> (batch_size, (neurons, inputs)), (batch_size, (neurons, inputs))
cell = tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu)
# agrega una capa para conectar las salidas de num_neurons a num_outputs
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

# training
# with tf.Session() as sess:
#     sess.run(init)
#
#     for iteration in range(num_train_iterations):
#
#         X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)
#         sess.run(train, feed_dict={X: X_batch, y: y_batch})
#
#         if iteration % 100 == 0:
#             mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
#             print(iteration, "\tMSE:", mse)
#
#     # Save Model for Later
#     saver.save(sess, "./rnn_time_series_model_j")

# predicting a time series t+1
# with tf.Session() as sess:
#     saver.restore(sess, "./rnn_time_series_model_j")
#
#     X_new = np.sin(np.array(train_inst[:-1].reshape(-1, num_time_steps, num_inputs)))
#     y_pred = sess.run(outputs, feed_dict={X: X_new})
#
# plt.title("Testing Model")
#
# # Training Instance
# plt.plot(train_inst[:-1], np.sin(train_inst[:-1]), "bo", markersize=15,alpha=0.5, label="Training Instance")
#
# # Target to Predict
# plt.plot(train_inst[1:], np.sin(train_inst[1:]), "ko", markersize=10, label="target")
#
# # Models Prediction
# plt.plot(train_inst[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
#
# plt.xlabel("Time")
# plt.legend()
# plt.tight_layout()
# plt.show()

# Generating new sequences
with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model_j")
    # SEED WITH ZEROS
    zero_seq_seed = [1.0 / (i + 1.0) for i in range(num_time_steps)]
    for iteration in range(len(ts_data.x_data) - num_time_steps):
        X_batch = np.array(zero_seq_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        zero_seq_seed.append(y_pred[0, -1, 0])

plt.plot(ts_data.x_data, zero_seq_seed, "b-")
plt.plot(ts_data.x_data[:num_time_steps], zero_seq_seed[:num_time_steps], "r", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

# with tf.Session() as sess:
#     saver.restore(sess, "./rnn_time_series_model_j")
#
#     # SEED WITH Training Instance
#     training_instance = list(ts_data.y_true[:30])
#     for iteration in range(len(training_instance) -num_time_steps):
#         X_batch = np.array(training_instance[-num_time_steps:]).reshape(1, num_time_steps, 1)
#         y_pred = sess.run(outputs, feed_dict={X: X_batch})
#         training_instance.append(y_pred[0, -1, 0])
#
# plt.plot(ts_data.x_data, ts_data.y_true, "b-")
# plt.plot(ts_data.x_data[:num_time_steps],training_instance[:num_time_steps], "r-", linewidth=3)
# plt.xlabel("Time")
# plt.show()