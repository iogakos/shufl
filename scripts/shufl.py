#!/usr/bin python

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

import pickle

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():

    #from http://stackoverflow.com/questions/18675863/load-data-from-python-pickle-file-in-a-loop
    def pickleLoader(f):
        try:
            while True:
                yield pickle.load(f)
        except EOFError:
            pass

    def load_mel_specs(filename):
        data = np.ndarray(21642 * 599 * 128, np.float64)
        data = data.reshape(-1, 1, 599, 128)

        with open(filename) as f:
            i = 0
            for mel in pickleLoader(f):
                data[i][0] = mel
                i += 1

        return data

    def load_tag_vectors(filename):
        data = np.ndarray((21642, 40), np.float32)
        with open(filename) as f:
            i = 0
            for tag_vector in pickleLoader(f):
                data[i] = tag_vector
                i += 1
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mel_specs('data/mels.pickle')
    y_train = load_tag_vectors('data/tags.pickle')

     # not ready for testing yet
#    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
#    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
#
    # We reserve the last 1000 training examples for validation.
    X_train, X_val = X_train[:-1000], X_train[-1000:]
    y_train, y_val = y_train[:-1000], y_train[-1000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val

# TODO: i think this is more like the bennanes network
def build_cnn(input_var=None):
    # Create a CNN following benanne's network: http://benanne.github.io/2014/08/05/spotify-cnns.html

    # Input layer - the spectrogram
    network = lasagne.layers.InputLayer(shape=(None, 1, 911, 128),
                                        input_var=input_var)

    # Convolutional layer with 256 filters of size 4 (time frames) (x128 or should it be 1?)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(4, 128),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.


    # Max-pooling layer of factor 4 in the time dimention:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(4, 1)) 

    # Another convolution with 256 kernels of size 4 in time dimension, and 2x1 pooling
    # (in time domain):
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(4, 1),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 1))

    # Another convolution with 256 4x1 kernels, and another 2x1 pooling
    # (in time domain):
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(4, 1),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 1))

    # Another convolution with 512 4x1 kernels:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=512, filter_size=(4, 1),
            nonlinearity=lasagne.nonlinearities.rectify)

    #TODO: need to figure out what now

    #extract mean and max from the last set of filters
    mean = lasagne.layers.GlobalPoolLayer(network, pool_function=theano.tensor.mean)
    maxx = lasagne.layers.GlobalPoolLayer(network, pool_function=theano.tensor.max)
    #l2 = lasagne.regularization.l2(network)

    #concatinate teh globally pooled layers
    concat = lasagne.layers.ConcatLayer([mean, maxx])


    # 2048-unit fully connected layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(concat, p=.5),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 2048-unit fully connected layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 40-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=40,
            nonlinearity=lasagne.nonlinearities.rectify)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=10):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')
    
    print("Building model and compiling functions...")
    network = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize, i.e mean square error):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)


    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy:
    test_acc = test_loss


    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])



    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 50, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 50, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))


    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 50, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))



if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [EPOCHS]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        main(**kwargs)
