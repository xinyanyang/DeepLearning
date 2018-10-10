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
num_inputs = 28
#number of outputs
num_outputs = 10
#the size of the filter k_y * k_x
filter_size = 5
#number of channels
num_channel = 3

#Initialize parameters K, W, b
model = {}
model['K'] = np.random.randn(num_channel,filter_size,filter_size) / 28
model['W'] = np.random.randn(num_outputs, num_inputs-filter_size+1, num_inputs-filter_size+1,num_channel) / 28
model['b'] = np.random.randn(num_outputs) / 28
model_grads = copy.deepcopy(model)

def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

def e(y, num_outputs=10):
    """
    e(y) function
    """
    ret = np.zeros(num_outputs, dtype=np.int)
    ret[y] = 1.0
    return ret  
    
def forward(x,y,model):
    x = x.reshape(num_inputs,num_inputs)
    Z = np.zeros((num_channel, num_inputs-filter_size+1, num_inputs-filter_size+1))
   
    for p in range(num_channel):
        for i in range(num_inputs-filter_size+1):
            for j in range(num_inputs-filter_size+1):
                mul_x = x[i:i+filter_size, j:j+filter_size]
                Z[p][i][j] = np.tensordot(mul_x, model['K'][p], axes=2)
    #implement the sigmoid activation function
    H = 1/(np.exp(-Z) + 1)
    H = H.reshape(num_inputs-filter_size+1, num_inputs-filter_size+1, num_channel)
    W_mul_H = np.zeros(num_outputs)
    for k in range(num_outputs):
        W_mul_H[k] = np.sum(np.multiply(model['W'][k], H))
    U = W_mul_H + model['b']
    p = softmax_function(U)
    return Z,H,p

def backward(x, y, p, H, Z, model, model_grads):
    x = x.reshape(num_inputs,num_inputs)
    dU = -e(y) + p
    delta = np.zeros((num_inputs-filter_size+1, num_inputs-filter_size+1,num_channel))
    for k in range(num_outputs):
        delta = delta + dU.reshape(num_outputs)[k]*model['W'][k]
    sigma_z_der = np.multiply(H,(1-H))
    sigmaz_delta = np.multiply(sigma_z_der, delta)
    sigmaz_delta = sigmaz_delta.reshape(num_channel,num_inputs-filter_size+1,num_inputs-filter_size+1)
    for p in range(num_channel):
        for i in range(filter_size):
            for j in range(filter_size):          
                mul_x2 = x[i:i+num_inputs-filter_size+1, j:j+num_inputs-filter_size+1]
                model_grads['K'][p][i][j] = np.tensordot(mul_x2, sigmaz_delta[p], axes=2)
    model_grads['b'] = dU
    for k in range(num_outputs):
        model_grads['W'][k] = dU.reshape(num_outputs)[k] * H
    return model_grads

#Training procedure
LR = 0.1
num_epochs = 10
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.01
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
        model['K'] = model['K'] - LR*model_grads['K']
        model['b'] = model['b'] - LR*model_grads['b']
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
