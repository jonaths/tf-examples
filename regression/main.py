import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


def basic_regression(x_data, y_true):
    my_data = pd.concat([pd.DataFrame(data=x_data, columns=['X Data']), pd.DataFrame(data=y_true, columns=['Y'])],
                        axis=1)

    print(my_data.head())

    # my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
    # plt.show()

    # Random 10 points to grab
    batch_size = 8

    # este guarda los datos que el programa va a modificar
    m = tf.Variable(0.5)
    b = tf.Variable(1.0)

    # con los placeholders se meten los datos de entrenamiento
    # se especifica el tipo y el shape
    xph = tf.placeholder(tf.float32, [batch_size])
    yph = tf.placeholder(tf.float32, [batch_size])

    # graph
    y_model = m * xph + b

    # loss function
    error = tf.reduce_sum(tf.square(yph - y_model))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(error)

    # initialize variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        batches = 1000
        for i in range(batches):
            # un arreglo de indices aleatorios
            rand_ind = np.random.randint(len(x_data), size=batch_size)
            # xph y yph tienen los elementos indexados por rand_ind
            feed = {xph: x_data[rand_ind], yph: y_true[rand_ind]}

            sess.run(train, feed_dict=feed)
        # aqui tengo las variables ahora
        # puedo recuperarlas corriendo  una sesion y regresando un arreglo con ellas
        model_m, model_b = sess.run([m, b])

    # ahora ya tengo las variables fuera de la sesion
    # toca recuperarlas para graficar y comparar

    print(model_m, model_b)
    y_hat = x_data * model_m + model_b
    my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
    plt.plot(x_data, y_hat, 'r')
    plt.show()


def using_estimator(x_data, y_true):
    # crear columnas de features. En este caso es una numerica que se llama x
    feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]
    # un estimador de tipo lineal que recibe la columna anterior como entrada, hay que darle la x
    estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

    # crea sets de entrenamiento y prueba
    x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size=0.3, random_state=101)
    print(x_train.shape, y_train.shape, x_eval.shape, y_eval.shape)

    # funciones para pasar al entrenar
    # con este lo entreno, por eso va con shuffle
    input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=4, num_epochs=None,
                                                    shuffle=True)
    # con este lo pruebo. Es el mismo arreglo
    train_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=4, num_epochs=1000,
                                                          shuffle=False)
    # Estos datos nunca los ha visto, Deberia de funcionar tambien
    eval_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_eval}, y_eval, batch_size=4, num_epochs=1000,
                                                         shuffle=False)

    estimator.train(input_fn=input_func, steps=1000)
    train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
    eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)

    print("train metrics: {}".format(train_metrics))
    print("eval metrics: {}".format(eval_metrics))


x_data = np.linspace(0, 10, 1000000)

noise = np.random.randn(len(x_data))

# y = mx + b + noise_levels
b = 5

y_true = (0.5 * x_data) + 5 + noise
# basic_regression(x_data, y_true)
using_estimator(x_data, y_true)
