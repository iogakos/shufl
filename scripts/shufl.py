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

mels_path = 'data/mels.pickle'
tags_path = 'data/tags.pickle'
mels_test_path = 'data/mels-test.pickle'
tags_test_path = 'data/tags-test.pickle'

# uncomment for running on aws
# data_root = '/mnt/'
# mels_path = data_root + mels_path
# tags_path = data_root + tags_path
# mels_test_path = data_root + mels_test_path
# tags_test_path = data_root + tags_test_path

# TODO: i think this is more like the bennanes network
def build_cnn(input_var=None):
    # Create a CNN following benanne's network: http://benanne.github.io/2014/08/05/spotify-cnns.html

    # Input layer - the spectrogram
    network = lasagne.layers.InputLayer(shape=(None, 1, 599, 128),
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

#from http://stackoverflow.com/questions/18675863/load-data-from-python-pickle-file-in-a-loop
def pickleLoader(f, batchsize):
    try:
        i = 0
        while True and i < batchsize:
            yield i, pickle.load(f)
            i +=1
    except EOFError:
        pass

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
def iterate_minibatches(totalsize, batchsize, mels_f, tags_f):
    for cnt in range(0, totalsize, batchsize):
        mel_data = np.ndarray(batchsize * 599 * 128, np.float16)
        mel_data = mel_data.reshape(-1, 1, 599, 128)

        tag_data = np.ndarray((batchsize, 40), np.float16)

        for i, mel in pickleLoader(mels_f, batchsize):
            mel_data[i][0] = mel

        for i, tag_vector in pickleLoader(tags_f, batchsize):
            tag_data[i] = tag_vector

        assert len(mel_data) == len(tag_data)

        yield mel_data, tag_data


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=10):
    # Load the dataset
    print("Loading data...")

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
    test_acc = 1 - test_loss

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

        mels_f = open(mels_path)
        tags_f = open(tags_path)

        train_err = 0
        train_batches = 0
        start_time = time.time()
        for inputs, targets in iterate_minibatches(800, 50, mels_f, tags_f):
            train_err += train_fn(inputs, targets)
            train_batches += 1
            print(train_err)

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for inputs, targets in iterate_minibatches(100, 50, mels_f, tags_f):
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

        mels_f.close()
        tags_f.close()

    mels_test_f = open(mels_test_path)
    tags_test_f = open(tags_test_path)
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(100, 10, mels_test_f, tags_test_f):
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
