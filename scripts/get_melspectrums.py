#!/bin/python

import os
import librosa
import sys

# just traverses the magna dataset mp3 samples and computes the mel spectrogram
# for each one of them. I have only adjusted the sample rate to the one magna's
# samples have been sampled and the default value of number of frequency bins
# is 128 so there is no need to adjust. Haven't yet looked at adjusting the
# overlapping window

magna_dir = 'magna'

dir_len = len(os.listdir(magna_dir))
dir_count = 1

for dirpath, directories, _ in os.walk(magna_dir):
  if dirpath == magna_dir: continue

  print 'dir: ', dir_count, '/', dir_len, '--', dirpath
  for subdir, _, filenames in os.walk(dirpath):

    file_count = 0
    for file in filenames:
      y, sr = librosa.load(os.path.join(subdir, file), sr = 16000)
      mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
      file_count += 1

      print '\r', 'done: ', file_count, '/', len(filenames), 
      sys.stdout.flush()
    print '\n',

    dir_count += 1
