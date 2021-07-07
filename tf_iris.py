import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_train_model(input_nodes, hidden_nodes, output_nodes, X_train, y_train, losses, number_of_iterations):

    tf.reset_default_graph()

    X = tf.placeholder(shape=(len(X_train), input_nodes), dtype=tf.float64, name="X")
    y = tf.placeholder(shape=(len(y_train), output_nodes), dtype=tf.float64, name="y")

    W1 = tf.Variable(np.random.rand(input_nodes, hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, output_nodes), dtype=tf.float64)

    H1 = tf.sigmoid(tf.matmul(X, W1))
    O = tf.sigmoid(tf.matmul(H1, W2))

    deltas = tf.square(O - y)
    loss = tf.reduce_sum(deltas)

    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(number_of_iterations):
        sess.run(train, feed_dict={X: X_train, y: y_train})
        losses[hidden_nodes].append(sess.run(loss, feed_dict={X: X_train, y: y_train}))

    weights1 = sess.run(W1)
    weights2 = sess.run(W2)

    print("loss (hidden nodes: {0}, iterations: {1}): {2:0.2f}".format(hidden_nodes, number_of_iterations,
                                                                      losses[hidden_nodes][-1]))

    sess.close()
    return weights1, weights2


iris_dataset = pd.read_csv("iris.csv")
X = np.array(iris_dataset[["sepal-length", "sepal-width", "petal-length", "petal-width"]]).reshape(-1, 4)
y = np.array(iris_dataset[["class"]]).reshape(-1, 1)

standart_scaler = StandardScaler()
X = standart_scaler.fit_transform(X)

dummy_encoder = OneHotEncoder(sparse=False, categories="auto")
y = dummy_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

input_nodes = 4
output_nodes = 3
number_of_iterations = 5000
plt.figure(figsize=(12, 8))

number_of_hidden_nodes = [1, 10, 30]
losses = {number_of_hidden_nodes[0]: [],
          number_of_hidden_nodes[1]: [],
          number_of_hidden_nodes[2]: []}

for hidden_nodes in number_of_hidden_nodes:
    create_train_model(input_nodes, hidden_nodes, output_nodes, X_train, y_train, losses, number_of_iterations)
    plt.plot(range(number_of_iterations), losses[hidden_nodes], label="ann: 4-{}-3".format(hidden_nodes))

plt.xlabel("Iterations", fontsize=12)
plt.ylabel("Loss", fontsize=12)

plt.show()
