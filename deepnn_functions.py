
import numpy as np
import h5py
import matplotlib.pyplot as plt


# Initialize shallow network parameters
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * .01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * .01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# Initialize deep network parameters
def initialize_parameters_deep(layer_dims):
    """
    return "W1", "b1", ..., "Wl", "bl":
        Wl - matrix shape (layer_dims[l], layer_dims[l-1])
        bl -- vector shape (layer_dims[l], 1)
    """

    parameters = {}
    num_layers = len(layer_dims)

    for l in range(1, num_layers):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * .01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
    assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


# Forward propagation
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    caches = []

    A = X

    num_layers = len(parameters) // 2

    # linear -> relu

    for l in range(1, num_layers):
        A_prev = A

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="relu")
        caches.append(cache)

    # linear -> sigmoid

    AL, cache = linear_activation_forward(A, parameters['W' + str(num_layers)], parameters['b' + str(num_layers)],
                                          activation="sigmoid")

    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = -(1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost


# Back propagation
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":

        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    num_layers = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)

    # Lth fully connected layer sigmoid activation backward
    current_cache = caches[num_layers - 1]
    grads["dA" + str(num_layers)], grads["dW" + str(num_layers)], grads[
        "db" + str(num_layers)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")

    # lth layer relu gradients
    for l in reversed(range(num_layers - 1)):
        current_cache = caches[l - 2]

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                    activation="relu")

        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# Update params
def update_parameters(parameters, grads, learning_rate):
    num_layers = len(parameters) // 2

    for l in range(1, num_layers + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters

def predict_binary_class(X, parameters):
    m = X.shape()
    n = len(parameters)
    predictions = np.zeros((1, m))

    ALs, caches = L_model_forward(X, parameters)

    for i in range(0, ALs.shape[1]):
        if ALs[0, i] > .5:
            predictions[0, i] = 1
        else: predictions[0, i] = 0


    return predictions

def pred_accuracy(X, Y, predictions):
    accuracy = np.sum((predictions == Y)/X.shape[1])

    return accuracy

# Activation
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))

    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert(dZ.shape == Z.shape)
    return dZ

def relu(Z):
    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0

    assert(dZ.shape == Z.shape)
    return dZ