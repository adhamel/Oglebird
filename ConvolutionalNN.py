# Convolutional Neural Network


import numpy as np
import matplotlib.pyplot as plt


np.random.seed(2818)


#### Zero-Padding

def zero_pad(X, pad):
    """
    shapes:
    padded image (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

    X_pad = np.pad(X, ((0,0), (pad, pad), (pad, pad), (0, 0)), 'constant')

    return X_pad



#### Convolution Step

def conv_single_step(a_slice_prev, W, b):

    """
    shapes:

    a_slice_prev (f, f, n_C_prev)
    W (f, f, n_C_prev)
    b (1, 1, 1)
    Z scalar
    """

    s = W * a_slice_prev    # Element-wise product of a_slice and W
    Z = np.sum(s)
    Z = Z + float(b)

    return Z

def conv_forward(A_prev, W, b, hparameters):

    """
    shapes:

    A_prev (m, n_H_prev, n_W_prev, n_C_prev)
    W (f, f, n_C_prev, n_C)
    b (1, 1, 1, n_C)
    Z (m, n_H, n_W, n_C)
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = int((n_H_prev - f + 2*pad)/stride + 1)
    n_W = int((n_W_prev - f + 2*pad)/stride + 1)

    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)

    # loop over training examples
    for i in range(m):

        a_prev_pad = A_prev_pad[i]

        for h in range(n_H):

            for w in range(n_W):

                for c in range(n_C):

                    # corners of slice
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f

                    # slice window of A_prev_pad
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # convolve
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

    assert(Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)

    return Z, cache



#### Forward Pooling

def pool_forward(A_prev, hparameters, mode = "max"):

    """
    shapes:

    A_prev (m, n_H_prev, n_W_prev, n_C_prev)
    A (m, n_H, n_W, n_C)
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):

        for h in range(n_H):

            for w in range(n_W):

                for c in range (n_C):

                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f

                    #print("vs:ve, hs:he", vert_start, vert_end, horiz_start, horiz_end)

                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # Pooling

                    if mode == "max":

                        A[i, h, w, c] = np.max(a_prev_slice)

                    elif mode == "average":

                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, hparameters)
    assert(A.shape == (m, n_H, n_W, n_C))

    return A, cache



#### Backpropagation

## Computing dA, dW, and db
def conv_backward(dZ, cache):

    """
    shapes:

    dZ (m, n_H, n_W, n_C)
    dA_prev (m, n_H_prev, n_W_prev, n_C_prev)
    dW (f, f, n_C_prev, n_C)
    db (1, 1, 1, n_C)
    """

    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros((A_prev.shape))
    dW = np.zeros((W.shape))
    db = np.zeros((b.shape))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):

        a_prev_pad = A_prev_pad[i]

        da_prev_pad = dA_prev_pad[i]



        for h in range(n_H):

            for w in range(n_W):

                for c in range(n_C):

                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]

                    dW[:,:,:,c] += a_slice*dZ[i, h, w, c]

                    db[:,:,:,c] += dZ[i, h, w, c]

        # Set the ith dA_prev to the unpadded da_prev_pad

        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db

## Backward pooling

# Max mask
def create_mask_from_window(x):

    """
    shapes:

    x (f, f)
    """

    mask = (x == np.max(x))

    return mask

# Average mask
def distribute_value(dz, shape):

    """
    shapes:

    dz scalar
    a (n_H, n_W)
    """

    (n_H, n_W) = shape

    average = dz/(n_H*n_W)

    a = np.ones(shape)*average

    return a

def pool_backward(dA, cache, mode = "max"):

    """
    shapes:

    dA - same shape as A
    dA_prev - same shape as A_prev
    """

    (A_prev, hparameters) = cache

    stride = hparameters['stride']
    f = hparameters['f']

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    dA_prev = np.zeros((A_prev.shape))

    for i in range(m):

        a_prev = A_prev[i]

        for h in range(n_H):

            for w in range(n_W):

                for c in range(n_C):

                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f

                    if mode == "max":

                        # current slice, mask
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                        mask = create_mask_from_window(a_prev_slice)

                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask*dA[i, h, w, c]

                    elif mode == "average":

                        shape = (f, f)

                        da = distribute_value(dA[i, h, w, c], shape)

                        # add distributed value of da
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += da

    assert(dA_prev.shape == A_prev.shape)

    return dA_prev
