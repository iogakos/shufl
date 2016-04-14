#!/binp/python

import gensim
import librosa

import pickle
import csv
import os
import numpy as np

model_path = 'data/d2vmodel.doc2vec'
tags_pickle_path = 'data/tags.pickle'
mels_pickle_path = 'data/mels.pickle'
tags_test_pickle_path = 'data/tags-test.pickle'
mels_test_pickle_path = 'data/mels-test.pickle'
tags_csv_path = 'data/annotations_final.csv'
magna_dir = 'data/magna'
# uncomment for aws
# data_root = '/mnt/'
# model_path = data_root + model_path
# tags_pickle_path = data_root + tags_pickle_path
# mels_pickle_path = data_root + mels_pickle_path
# tags_test_pickle_path = data_root + tags_test_pickle_path
# mels_test_pickle_path = data_root + mels_test_pickle_path
# magna_dir = data_root + magna_dir

# prepare traverse of tags list
tags_csv_file = open(tags_csv_path, 'r')
reader = csv.reader(tags_csv_file, delimiter='\t')
tags_list = list(reader)

# prepare tags pickle file
tags_pickle_file = open(tags_pickle_path, 'w')
mels_pickle_file = open(mels_pickle_path, 'w')

tags_test_pickle_file = open(tags_test_pickle_path, 'w')
mels_test_pickle_file = open(mels_test_pickle_path, 'w')

# load tags model
model = gensim.models.Doc2Vec.load(model_path)

# mel spec properties
stft_window = 3116
hop = stft_window / 4

count = 0
for row in tags_list[1:]:
    clip_id = 'CLIP_' + row[0]

    # model does not support this clip_id, check the next one
    if not model.docvecs.doctags.has_key(clip_id): continue

    # extract mel spec from sample
    sample_filename = os.path.join(os.getcwd(), magna_dir, row[-1])
    y, sr = librosa.load(sample_filename, sr=16000)
    spectrum = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmax=sr/2, n_fft = stft_window,
            hop_length=hop)

    # this gives us 10 decimal point precision, so ~-80 dB.
    # should be enough and saves a lot of space
    spectrum = spectrum.astype(np.float16)
    #flip frequency and time axis
    spectrum = spectrum.transpose()

    tags_vector = model.docvecs[clip_id].astype(np.float16)

    # Write nparrays in pickle files. For each file, its tag vector and mel
    # spec nparrays MUST be in the same line number on the train files

    # TODO: if we want to save 10% for testing we could just take every 10th
    # sample for testing - it should be more evenly distributed as sometimes
    # there are multiple parts of the same song one after another
    # if count%10 == 0:
    if count >= 900:
        pickle.dump(tags_vector, tags_test_pickle_file)
        pickle.dump(spectrum, mels_test_pickle_file)
    else:
        pickle.dump(tags_vector, tags_pickle_file)
        pickle.dump(spectrum, mels_pickle_file)

    count += 1
    print '\r', 'done: ', count, '/', model.docvecs.count, \
            '(', count * 100 / model.docvecs.count, '%)',

    if count == 1000: break

tags_pickle_file.close()
mels_pickle_file.close()
tags_test_pickle_file.close()
mels_test_pickle_file.close()
