# Exploring Tensorflow

import math
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
from PIL import Image
from scipy import ndimage

from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict



# Linear function
def linear_function():

    """

    Implements a linear function:

            Initializes W to be a random tensor of shape (4,3)

            Initializes X to be a random tensor of shape (3,1)

            Initializes b to be a random tensor of shape (4,1)

    Returns:

    result -- runs the session for Y = WX + b

    """

    np.random.seed(2818)

    X = tf.constant(np.random.randn(3, 1), name = "X")

    W = tf.constant(np.random.randn(4, 3), name = "W")

    b = tf.constant(np.random.randn(4, 1), name = "b")

    Y = tf.add(tf.matmul(W, X), b)


    sess = tf.Session()
    result = sess.run(Y)

    sess.close()

    return result


# Sigmoid
def sigmoid(z):

    x = tf.placeholder(tf.float32, name = "x")

    sigmoid = tf.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict = {x: z})

    return result


# Cost
def cost(logits, labels):

    z = tf.placeholder(tf.float32, name = "z")
    y = tf.placeholder(tf.float32, name = "y")

    cost = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = z)

    sess = tf.Session()
    cost = sess.run(cost, feed_dict = {z: logits, y:labels})

    sess.close()

    return cost


# One-hot matrix for classes
def one_hot_matrix(labels, C):

    C = tf.constant(C, name = "C")

    one_hot_matrix = tf.one_hot(labels, C, axis = 0)

    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)

    sess.close()

    return one_hot
# non-tensorflow implementation
def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)].T

    return Y


# Mini-batching
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):

    m = X.shape[1]
    mini_batches = []
    np.random.seed(seed)

    # shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # partition
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # last batch if not equal batch size
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# Initialize
def ones(shape):

    ones = tf.ones(shape)

    sess = tf.Session()
    ones = sess.run(ones)

    sess.close()

    return ones


# Load data
train_dir = 'input/train'
test_dir = 'input/test'

labels = pd.read_csv('train_labels.csv')
n = len(labels)
classes = set(labels['class'])
n_class = len(classes)
class_num = dict(zip(classes, range(n_class)))
width = 64

X_train_orig = np.zeros((n, width, width, 3), dtype=np.uint8)
train_y = np.zeros((n, n_class), dtype=np.uint8)

for i in tqdm(range(n)):
    X_train_orig[i] = cv2.resize(cv2.imread('input/train/%s.jpg' % labels['id'][i]), (width, width))
    train_y[i][class_num[labels['class'][i]]] = 1

test_labels = pd.read_csv('test_labels.csv')
n_test = len(test_labels)

X_test_orig = np.zeros((n_test, width, width, 3), dtype=np.uint8)
test_y = np.zeros((n_test, n_class), dtype=np.uint8)

for i in tqdm(range(n_test)):
    X_test_orig[i] = cv2.resize(cv2.imread('input/test/%s.jpg' % test_labels['id'][i]), (width, width))
    test_y[i][class_num[test_labels['class'][i]]] = 1

# Flatten and Normalize

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X_train = (X_train_flatten - X_train_flatten.mean()) / (X_train_flatten.std() + 1e-8)
X_test = (X_test_flatten - X_test_flatten.mean()) / (X_test_flatten.std() + 1e-8)

# Convert training and test labels to one hot matrices
Y_train = one_hot_matrix(train_y, n_class)
Y_test = one_hot_matrix(test_y, n_class)


# Create Tensorflow session placeholders
def create_placeholders(n_x, n_y):

    """
    shapes:

    n_x scalar size of image vector (num_px * num_px * 3)
    n_y scalar, number of classes (from 0 to 5, so -> 6)
    X data input placeholder [n_x, None], data type "float"
    Y input labels placeholder [n_y, None], data type "float"
    """

    # in shape, None leaves number of train/test examples flexible
    X = tf.placeholder(shape = [n_x, None], dtype = "float", name = "X")

    Y = tf.placeholder(shape = [n_y, None], dtype = "float", name = "Y")

    return X, Y


# Initialize Network Parameters
"""
architecture W1, b1, W2, b2, W3, b3
shapes:
    
    W1 : [25, 12288]    num_px * num_px * 3 = 64*64*3
    b1 : [25, 1]
    W2 : [12, 25]
    b2 : [12, 1]
    W3 : [6, 12]
    b3 : [6, 1]
"""
def initialize_parameters():

    W1 = tf.get_variable("W1", [25, 12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [6, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


# Forward Prop
def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                                              # Z3 = np.dot(W3,Z2) + b3

    return Z3

tf.reset_default_graph()

def compute_cost(Z3, Y):

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    return cost

tf.reset_default_graph()


# Back Prop

# 3-layer tensorflow network
# linear -> relu -> linear -> relu -> linear -> softmax

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):

    # setup
    ops.reset_default_graph()   # to rerun the model without overwriting tf variables

    tf.set_random_seed(2818)
    seed = 0                    # seed for minibatches

    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]

    costs = []


    # create placeholders, initialize parameters
    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters()


    # forward prop in tensor graph
    Z3 = forward_propagation(X, parameters)


    # Cost
    cost = compute_cost(Z3, Y)


    # Back Prop
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)


    # Initialize all variables
    init = tf.global_variables_initializer()


    # Start session to compute tensorflow graph
    with tf.Session() as sess:

        # initialize
        sess.run(init)

        # train
        for epoch in range(num_epochs):

            epoch_cost = 0.

            num_minibatches = int(m / minibatch_size)

            seed = seed + 1

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                # run the graph on minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches


            # cost after each epoch (implementation note: modulo % returns remainder)
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("learning rate: " + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)

        print ("parameters trained.")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))


        return parameters


# Prediction
def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters['W1'])
    b1 = tf.convert_to_tensor(parameters['b1'])
    W2 = tf.convert_to_tensor(parameters['W2'])
    b2 = tf.convert_to_tensor(parameters['b2'])
    W3 = tf.convert_to_tensor(parameters['W3'])
    b3 = tf.convert_to_tensor(parameters['b3'])

    params = {'W1': W1,
              'b1': b1,
              'W2': W2,
              'b2': b2,
              'W3': W3,
              'b3': b3}

    x = tf.placeholder(dtype="float", shape = [12288, 1])

    z3 = forward_propagation(x, params)
    p = tf.arg_max(z3)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})

    return prediction

def single_image_predict(path, filename, parameters):
    fname = str(path) + str(filename)
    image = np.array(ndimage.imread(fname, flatten=False))
    image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
    prediction = predict(image, parameters)

    plt.imshow(image)
    print("algorithm predicts y = " + str(np.squeeze(prediction)))

    return prediction


