#!/binp/python

import gensim
import librosa

import pickle
import csv
import os
import numpy as np

import argparse
parser = argparse.ArgumentParser(
        description='GI15 2016 - Prepare dataset for training/validation')
parser.add_argument('-c', '--clips-only', action='store_true',
        help='Creates a line separated file with the CLIP_IDs')
parser.add_argument('-p', '--production', action='store_true',
	help='Generates dataset appropriate for the production instance')

args = parser.parse_args()

model_path = 'data/d2vmodel.doc2vec'
tags_pickle_path = 'data/tags.pickle'
mels_pickle_path = 'data/mels.pickle'
tags_test_pickle_path = 'data/tags-test.pickle'
mels_test_pickle_path = 'data/mels-test.pickle'
tags_csv_path = 'data/annotations_final.csv'
magna_dir = 'data/magna'
clips_path = 'data/clips'

# prepare traverse of tags list
tags_csv_file = open(tags_csv_path, 'r')
reader = csv.reader(tags_csv_file, delimiter='\t')
tags_list = list(reader)

if args.clips_only is False:
    # prepare tags pickle file
    tags_pickle_file = open(tags_pickle_path, 'w')
    mels_pickle_file = open(mels_pickle_path, 'w')

    tags_test_pickle_file = open(tags_test_pickle_path, 'w')
    mels_test_pickle_file = open(mels_test_pickle_path, 'w')
clips_file = open(clips_path, 'w')

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

    clips_file.write(''.join(s for s in [clip_id, '\n']))
    if args.clips_only is True: continue

    try:
        y, sr = librosa.load(sample_filename, sr=16000)
    except IOError:
        print 'file not found: ' + row[-1]
    except EOFError:
        print 'file broken: ' + row[-1]
    else:

        spectrum = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, fmax=sr/2, n_fft = stft_window,
                hop_length=hop)

        # this gives us 10 decimal point precision, so ~-80 dB.
        # should be enough and saves a lot of space
        spectrum = spectrum.astype(np.float32)
        #flip frequency and time axis
        spectrum = spectrum.transpose()

        tags_vector = model.docvecs[clip_id]
        tags_vector = np.add(tags_vector, np.abs(tags_vector.min()))

        # Write nparrays in pickle files. For each file, its tag vector and mel
        # spec nparrays MUST be in the same line number on the train files

        # TODO: if we want to save 10% for testing we could just take every
        # 10th sample for testing - it should be more evenly distributed as
        # sometimes there are multiple parts of the same song one after another
        # if count%10 == 0:
        if count >= 20000:
            pickle.dump(tags_vector, tags_test_pickle_file)
            pickle.dump(spectrum, mels_test_pickle_file)
        else:
            pickle.dump(tags_vector, tags_pickle_file)
            pickle.dump(spectrum, mels_pickle_file)

        count += 1
        print '\r', 'done: ', count, '/', model.docvecs.count, \
                '(', count * 100 / model.docvecs.count, '%)',

	if args.production is False and count == 1000: break

if args.clips_only is False:
    tags_pickle_file.close()
    mels_pickle_file.close()
    tags_test_pickle_file.close()
    mels_test_pickle_file.close()
clips_file.close()
