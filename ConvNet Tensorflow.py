
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *


# Create Tensorflow session placeholders
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    shapes:

    X data input placeholder [None, n_H0, n_W0, n_C0], datatype "float"
    Y input labels placeholder [None, n_y], datatype "float"
    """

    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = "X")

    Y = tf.placeholder(tf.float32, [None, n_y], name = "Y")

    return X, Y

def initialize_parameters():

    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters



# Forward prop
def forward_propagation(X, parameters):


    W1 = parameters['W1']
    W2 = parameters['W2']

    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')

    Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

    # flatten for fully-connected layer

    P2 = tf.contrib.layers.flatten(P2)

    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn = None)

    return Z3



# Cost
def compute_cost(Z3, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))

    return cost



"""
Example implementation:

tf.reset_default_graph()

with tf.Session() as sess:

    np.random.seed(1)

    X, Y = create_placeholders(64, 64, 3, 6)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    init = tf.global_variables_initializer()

    sess.run(init)

    a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})

    print("cost = " + str(a))
"""



def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009, num_epochs = 100, minibatch_size = 64, print_cost = True):

    # setup
    ops.reset_default_graph()
    tf.set_random_seed(2818)
    seed = 0

    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]

    costs = []

    # create placeholders, initialize parameters
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    parameters = initialize_parameters()


    # forward prop in tensor graph
    Z3 = forward_propagation(X, parameters)


    # cost
    cost = compute_cost(Z3, Y)

    # back prop
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)


    # initialize variables
    init = tf.global_variables_initializer()



    # Start session to compute tensorflow graph
    with tf.Session() as sess:

        # initialize
        sess.run(init)

        # train
        for epoch in range(num_epochs):

            epoch_cost = 0.

            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set

            seed = seed + 1

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                # run the graph on minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches


            # cost after each epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)


        return train_accuracy, test_accuracy, parameters



# _, _, parameters = model(X_train, Y_train, X_test, Y_test)
