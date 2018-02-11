## CNN modeling with Keras API and Tensorflow Framework
##
## Convolutional NNs from deeplearning.ai

# Purpose: an algorithm to classify faces as happy/not happy


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors

X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


def HappyModel(input_shape):

    X_input = Input(input_shape)
    X = ZeroPadding2D((3,3))(X_input)

    X = Conv2D(16, (3, 3), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name = 'max_pool0')(X)

    X = Conv2D(32, (5, 5), strides = (1, 1), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name = 'max_pool1')(X)

    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name = 'max_pool2')(X)

    X = Flatten()(X)
    X = Dense(128, activation = 'relu', name = 'fc1')(X)
    X = Dense(1, activation = 'sigmoid', name = 'fc2')(X)

    model = Model(inputs = X_input, outputs = X, name = 'HappyModel')

    return model


# Implementing Keras steps:
#       create, compile, train, test

# create model
happyModel = HappyModel((64, 64, 3))

# compile model, define arguments based on classification problem
happyModel.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ["accuracy"])

# train model
happyModel.fit(x = X_train, y = Y_train, epochs = 10, batch_size = 16)

## test/evaluate model.
preds = happyModel.evaluate(x = X_test, y = Y_test)

print ("Loss: " + str(preds[0]))
print ("Test Accuracy: " + str(preds[1]))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(happyModel.predict(x))

happyModel.summary()  # print layer details
plot_model(happyModel, to_file='HappyModel.png')  # plot graph

SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))
