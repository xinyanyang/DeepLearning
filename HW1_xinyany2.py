#coding: utf-8
import numpy as np
import h5py
import time
import copy
from random import randint
import random

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

#Implementation of stochastic gradient descent algorithm
#number of inputs
num_inputs = 28*28
#number of outputs
num_outputs = 10
num_hidden = 100

#Initialize parameters W, C, b_1, b_2
model = {}
model['W'] = np.random.randn(num_hidden, num_inputs) / np.sqrt(num_inputs)
model['C'] = np.random.randn(num_outputs, num_hidden) / np.sqrt(num_inputs)
model['b_1'] = np.random.randn(num_hidden)
model['b_2'] = np.random.randn(num_outputs)
model_grads = copy.deepcopy(model)

def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

def ReLU(x):
    """
    Implement the RELU function
    """
    return x * (x > 0)

def e(y, num_outputs=10):
    """
    e(y) function
    """
    ret = np.zeros(num_outputs, dtype=np.int)
    ret[y] = 1.0
    return ret

def dReLU(z):
    """
    derivative of ReLU function
    """
    # To avoid bug when the overflow happens
    if z - 0.0 == 0:
        z = z + 1e-8
    if z < 0:
        return 0.0
    else:
        return 1.0

def forward(x,y,model):
    Z = np.dot(model['W'],x) + model['b_1']
    H = np.array([ReLU(z) for z in Z])
    U = np.dot(model['C'], H) + model['b_2']
    p = softmax_function(U)
    return Z,H,p

def backward(x, y, p, H, Z, model, model_grads):
    dU = -e(y) + p
    rou_b_2 = dU
    rou_C = np.dot(dU.reshape(num_outputs, 1), H.reshape(1, num_hidden))
    delta = np.dot(np.transpose(model['C']), dU)
    rou_b_1 = delta * [dReLU(z) for z in Z]
    rou_w = np.dot(rou_b_1.reshape(len(rou_b_1), 1), np.transpose(x.reshape(len(x), 1)))
        
    model_grads['C'] = rou_C
    model_grads['b_2'] = rou_b_2
    model_grads['W'] = rou_w
    model_grads['b_1'] = rou_b_1
    assert model_grads['C'].shape == rou_C.shape
    assert model_grads['b_2'].shape == rou_b_2.shape
    assert model_grads['W'].shape == rou_w.shape
    assert model_grads['b_1'].shape == rou_b_1.shape
    return model_grads

LR = 0.01
num_epochs = 12
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
    total_correct = 0
    for n in range(len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        Z,H,p = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x, y, p, H, Z, model, model_grads)
        #update parameters
        model['C'] = model['C'] - LR*model_grads['C']
        model['b_2'] = model['b_2'] - LR*model_grads['b_2']
        model['b_1'] = model['b_1'] - LR*model_grads['b_1']
        model['W'] = model['W'] - LR*model_grads['W']
        
    print('epochs: '+ str(epochs), ' | Training Accuracy: ' + str(total_correct/np.float(len(x_train))))



#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    Z,H,p = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print('Test Accuracy:', total_correct/np.float(len(x_test)))