## Binary Classification Model

import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from deepnn_functions import *
import pandas as pd

# plot parameters, set seed, load data
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(2418)

# Load data
train_dir = 'input/train'
test_dir = 'input/test'

labels = pd.read_csv('train_labels.csv')  # labels for training set
n = len(labels)
classes = set(labels['class'])
n_class = len(classes)
class_num = dict(zip(classes, range(n_class)))
width = 299

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


## Exploratory

# View an indexed image
index = 30
plt.imshow(X_train_orig[index])
print ("y = " + str(train_y[0,index]) + ". Class label: " + classes[train_y[0,index]].decode("utf-8"))

# parameters
m_train = X_train_orig.shape[0]
num_px = X_train_orig.shape[1]
m_test = X_test_orig.shape[0]

print ("m_train: " + str(m_train))
print ("m_test: " + str(m_test))
print ("image size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(X_train_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(X_test_orig.shape))
print ("test_y shape: " + str(test_y.shape))

#Image to vector conversion.
# Reshape train and test original sets

train_x_flatten = X_train_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = X_test_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize flattened train and test sets

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print("train_x shape:" + str(train_x.shape))
print("test_x shape: " + str(test_x.shape))


## Shallow Model Constants

n_x = num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)


# Shallow Network

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):

    # Initialization

    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]


    # Gradient Descent

    for i in range(0, num_iterations):

        # Forward prop
        A1, cache1 = linear_activation_forward(X, W1, b1, activation = "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation = "sigmoid")

        # Cost
        cost = compute_cost(A2, Y)

        # L layer backprop
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Back prop
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation = "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation = "relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update params
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Cost after each 100 examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # Plot cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)

pred_train = predict_binary_class(train_x, parameters)
pred_test = predict_binary_class(test_x, parameters)

accuracy_train = pred_accuracy(train_x, train_y, pred_train)
accuracy_test = pred_accuracy(test_x, test_y, pred_test)



## Deep Model Architecture
layers_dims = [n_x, 20, 7, 5, 1]


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009

    # Initialization.

    costs = []
    parameters = initialize_parameters_deep(layers_dims)


    # Gradient Descent

    for i in range(0, num_iterations):

        # Forward prop
        AL, caches = L_model_forward(X, parameters)

        # Cost
        cost = compute_cost(AL, Y)

        # Back prop
        grads = L_model_backward(AL, Y, caches)

        # Update params
        parameters = update_parameters(parameters, grads, learning_rate)

        # Cost after each 100 examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # Plot cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


# Train 5-layer network

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
pred_train = predict_binary_class(train_x, parameters)
pred_test = predict_binary_class(test_x, parameters)

accuracy_train = pred_accuracy(train_x, train_y, pred_train)
accuracy_test = pred_accuracy(test_x, test_y, pred_test)



def pred_image(path, parameters):
    image = np.array(ndimage.imread(path, flatten=False))
    input_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
    prediction = predict_binary_class(input_image, parameters)

    plt.imshow(image)
    return prediction


