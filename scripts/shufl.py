#!/usr/bin python

from __future__ import print_function

import sys
import os
import time
import ConfigParser

import numpy as np
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

import lasagne
import gensim

import pickle
import argparse

config_path = 'config/shufl.cfg'

tags_file = 'data/tags'
mels_path = 'data/mels.pickle'
tags_path = 'data/tags.pickle'
clips_path = 'data/clips'
mels_test_path = 'data/mels-test.pickle'
tags_test_path = 'data/tags-test.pickle'

model_path = 'data/shufl.pickle'
d2v_model_path = 'data/d2vmodel.doc2vec'

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
            W=lasagne.init.Normal())
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
            nonlinearity=lasagne.nonlinearities.softmax)

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
def it_minibatches(totalsize, batchsize, mels_f, tags_f, shuffle=False,
        inputs=None, targets=None):

    if shuffle is True:
         
        assert len(inputs) == len(targets)

        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    else:
        for cnt in range(0, totalsize, batchsize):
            mel_data = np.ndarray(batchsize * 599 * 128, np.float32)
            mel_data = mel_data.reshape(-1, 1, 599, 128)

            tag_data = np.ndarray((batchsize, 40), np.float32)

            for i, mel in pickleLoader(mels_f, batchsize):
                mel_data[i][0] = mel

            for i, tag_vector in pickleLoader(tags_f, batchsize):
                tag_data[i] = tag_vector

            assert len(mel_data) == len(tag_data)

            yield mel_data, tag_data


def validate(val_fun, config={}):
    # open pickle files for the test set
    mels_test_f = open(mels_test_path)
    tags_test_f = open(tags_test_path)

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for inputs, targets in it_minibatches(
            config['test_data'],
            config['test_batchsize'],
            mels_f,
            tags_f,
            shuffle=shuffle,
            inputs=mels_test_mem, targets=tags_test_mem):

        inputs, targets = batch
        err, acc, _ = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    mels_test_f.close()
    tags_test_f.close()

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=200, mode='train', track_id=None, checkpoint=True,
        production=False, shuffle=False, config={}):

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
            loss, params, learning_rate=0.001, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy:
    test_acc = 1 - test_loss

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function(
            [input_var, target_var], [test_loss, test_acc, prediction])

    if mode == 'train':
        print("Entered training mode")

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # Finally, launch the training loop.
	if checkpoint is True: #load previously trained model
            print("Continuing training from last epoch. Loading model..")
            with open(model_path, 'r') as f:
                params = pickle.load(f)

            lasagne.layers.set_all_param_values(network, params)
        else: # start training from scratch
            print("Starting training from scratch...")

        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:

            # open pickle files for the train/validation set
            mels_f = open(mels_path)
            tags_f = open(tags_path)

            train_err = 0
            train_batches = 0
            start_time = time.time()
            for inputs, targets in it_minibatches(
                    int(config['train_data']),
                    int(config['train_batchsize']),
                    mels_f,
                    tags_f,
                    shuffle=shuffle,
                    inputs=mels_mem, targets=tags_mem):

                train_err += train_fn(inputs, targets)
                train_batches += 1
                print(train_err/train_batches)

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for inputs, targets in it_minibatches(
                    int(config['val_data']),
                    int(config['val_batchsize']),
                    mels_f,
                    tags_f,
                    shuffle=shuffle,
                    inputs=mels_val_mem, targets=tags_val_mem):

                err, acc, _ = val_fn(inputs, targets)
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

            # close pickle files to restart the stream for pickle loader
            mels_f.close()
            tags_f.close()

            print("Saving model...")
            params = lasagne.layers.get_all_param_values(network)
            with open(model_path, 'w') as f:
                pickle.dump(params, f)

        validate(val_fn, config)
    elif mode == 'val':
        print("Entered validation mode")
        with open(model_path, 'r') as f:
            params = pickle.load(f)

        lasagne.layers.set_all_param_values(network, params)
        validate(val_fn, config)
    else:
        print("Entered user mode")
        if track_id is not None:
            print("Calculating closest vectors for %s" % track_id)
            with open(model_path, 'r') as f:
                params = pickle.load(f)


            with open(clips_path, 'r') as f:
                found = False
                for line_no, line in enumerate(f, 0):
                    if track_id == line[:-1]:
                        found = True
                        break

            if found is True:
                spectrogram = np.ndarray(
                        int(config['mel_x']) * int(config['mel_y']), np.float32)
                spectrogram = spectrogram.reshape(
                        -1, 1, int(config['mel_x']), int(config['mel_y']))
                
                found = False
                with open(mels_path) as mels_f:
                    for i, mel in pickleLoader(mels_f, int(config['data'])):
                        if i == line_no:
                            found = True
                            spectrogram[0] = mel
                            break

                if found is False:
                    print("Could not find cached mel spectrum for %s" % \
                            track_id)
                else:
                    lasagne.layers.set_all_param_values(network, params)

                    tag_zeros = np.zeros(
                            (1, int(config['latent_v'])), np.float32)
                    _, _, tag_prediction = val_fn(spectrogram, tag_zeros)
                    d2v_model = gensim.models.Doc2Vec.load(d2v_model_path)

                    # find minimum value among all vectors
                    minn = np.amin(np.array([d2v_model.docvecs[c] for c in d2v_model.docvecs.doctags.keys()]).flatten())

                    # get the real representation and modify according to our
                    # model i.e. by elementwise adding global minimum
                    real = d2v_model.docvecs[args.track_id]
                    real = np.add(real, np.abs(minn))

                    print('predict\t  real\t    diff')
                    for idx, v in np.ndenumerate(tag_prediction[0]):
                        print("{:1.6f}".format(v) + "{:10.6f}".format(real[idx]) + "{:10.6f}".format(v-real[idx]))

                    #create clip_id->tags dictionary
                    tags_dict = dict()
                    with open(tags_file, "r") as f:
                        for index, line in enumerate(f):
                            tags = line.split()
                            tags_dict[tags[0]] = tags[1:]

                    # iterate over the d2v dictionary and find the maximally
                    # similar songs by calculating the euclidian distance to
                    # each
                    values = [
                        tuple([
                            np.linalg.norm(
                            tag_prediction-np.add(d2v_model.docvecs[clip], np.abs(minn))),clip])
                            for clip in d2v_model.docvecs.doctags.keys()]

                    arr = np.array(
                            values, dtype=[('dist','float32'),('id','|S10')])
                    inds = np.argsort(arr['dist'])
                    top10 = inds[:10]
                    result = np.ndarray(
                            (10,), dtype=[('dist','float32'),('id','|S10')])

                    np.take(arr, top10, out=result)
                    for r in result['id']:
                        print(r + ' ' + str(tags_dict[r[5:]]).replace(' ','').replace('.','').replace('[','').replace(']',''))
            else:
                print(track_id, " not found")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GI15 2016 - Shufl')
    parser.add_argument('-e', '--epochs', metavar='N', type=int,
            help='Number of training epochs (default: 500)')
    parser.add_argument('-m', '--mode', metavar='<mode>',
            help='Set to `train` for training, `val` for validating. ' + \
                    '`user` for user mode (Default `train`)')
    parser.add_argument('-t', '--track-id',
            help='Return the closest vectors for a specific track id from ' + \
                    'the MagnaTagATune dataset')
    parser.add_argument('-c', '--checkpoint', action='store_true',
            help='Continue training from the last epoch checkpoint')
    parser.add_argument('-p', '--production', action='store_true',
            help='Load training specific options appropriate for production')
    parser.add_argument('-s', '--shuffle', action='store_true',
            help='Shuffle the training data (caution: loads the whole ' + \
                    'datas in memory)')


    config = ConfigParser.RawConfigParser()
    config.read(config_path)

    args = parser.parse_args()
    if args.production is True:
        print('Loading production configuration')
        cfg = config._sections['production']
    else:
        print('Loading local configuration')
        cfg = config._sections['local']

    mels_mem = mels_val_mem = mels_test_meme = \
        tags_mem = tags_val_mem = tags_test_mem = None

    kwargs = {}
    if args.epochs is not None:
        kwargs['num_epochs'] = args.epochs
    if args.mode is not None:
        kwargs['mode'] = args.mode
    if args.track_id is not None:
        kwargs['track_id'] = args.track_id
    if args.checkpoint is not None:
        kwargs['checkpoint'] = args.checkpoint
    if args.production is not None:
        kwargs['config'] = args.production
    if args.shuffle is not None:
        kwargs['shuffle'] = args.shuffle
    if args.shuffle is True:
        mels_f = open(mels_path)
        tags_f = open(tags_path)

        mels_mem = np.ndarray(int(cfg['train_data']) * 599 * 128, np.float32)
        mels_mem = mels_mem.reshape(-1, 1, 599, 128)

        mels_val_mem = np.ndarray(int(cfg['val_data']) * 599 * 128, np.float32)
        mels_val_mem = mels_val_mem.reshape(-1, 1, 599, 128)

        mels_test_mem = np.ndarray(int(cfg['test_data']) * 599 * 128, np.float32)
        mels_test_mem = mels_test_mem.reshape(-1, 1, 599, 128)

        tags_mem = np.ndarray((int(cfg['train_data']), 40), np.float32)
        tags_val_mem = np.ndarray((int(cfg['val_data']), 40), np.float32)
        tags_test_mem = np.ndarray((int(cfg['test_data']), 40), np.float32)

        for i, mel in pickleLoader(mels_f, int(cfg['train_data'])):
            mels_mem[i][0] = mel
        for i, mel in pickleLoader(mels_f, int(cfg['val_data'])):
            mels_val_mem[i][0] = mel
        for i, mel in pickleLoader(mels_f, int(cfg['test_data'])):
            mels_test_mem[i][0] = mel

        for i, tag_vector in pickleLoader(tags_f, int(cfg['train_data'])):
            tags_mem[i] = tag_vector
        for i, tag_vector in pickleLoader(tags_f, int(cfg['val_data'])):
            tags_val_mem[i] = tag_vector
        for i, tag_vector in pickleLoader(tags_f, int(cfg['test_data'])):
            tags_test_mem[i] = tag_vector

        print(mels_mem.shape)
        print(mels_val_mem.shape)
        print(mels_test_mem.shape)
        print(tags_mem.shape)
        print(tags_val_mem.shape)
        print(tags_test_mem.shape)

        mels_f.close()
        tags_f.close()


    kwargs['config'] = cfg

    main(**kwargs)
