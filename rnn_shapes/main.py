import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell

# numero de neuronas
dim = 12

# el primer elemento es el tamano del batch
x = tf.placeholder(tf.float32, shape=[None, dim])
y = tf.placeholder(tf.float32, shape=[4, dim])
z = tf.placeholder(tf.float32, shape=[None, dim + 1])
print('x, y, z:', x.shape, y.shape, z.shape)

# exp1: sin ProjectionWrapper
print("exp1")

cell = BasicRNNCell(dim)
state1 = cell.zero_state(batch_size=4, dtype=tf.float32)
state2 = cell.zero_state(batch_size=8, dtype=tf.float32)

# output, state
out1, out2 = cell(x, state1)
print(out1.shape, out2.shape)

out1, out2 = cell(x, state2)
print(out1.shape, out2.shape)

out1, out2 = cell(y, state1)
print(out1.shape, out2.shape)

# exp2: con ProjectionWrapper
print("exp2")

cell = BasicRNNCell(dim)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=2)
state1 = cell.zero_state(batch_size=4, dtype=tf.float32)
state2 = cell.zero_state(batch_size=8, dtype=tf.float32)

# output, state
out1, out2 = cell(x, state1)
print(out1.shape, out2.shape)

out1, out2 = cell(x, state2)
print(out1.shape, out2.shape)

out1, out2 = cell(y, state1)
print(out1.shape, out2.shape)