import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import h5py

dataset_path = '/Users/wiktor/Documents/Projects/IRDM/shufl/magnatagatune_dataset/mp3/'

tags_file = 'data/tags'
idToPath_file = 'data/idToPath'

idToPath = {}
stft_window = 2048
hop = stft_window / 4

#create h5py file for storing spectrogram arrays
spectrograms_file = h5py.File('data/spectrograms.hdf5','a')
#debug: compression tests
# spectrograms_file_comp = h5py.File('data/spectrograms_comp.hdf5','a')

#create clip_id <-> filename mappings
for line in open(idToPath_file):
  tags = line.split()
  idToPath[tags[0]] = tags[1]

total_clips = 25863

#debug: compression tests
# test = 0

# generate spectogram for each clip with non empty tags
with open(tags_file, "r") as f:
	for index, line in enumerate(f):
	  tags = line.split()
	  if (len(tags) > 0):
	  	path = dataset_path+idToPath[tags[0]]
		
		try:
			y, sr = librosa.load(path, sr=16000)
		except IOError:
			print idToPath[tags[0]] + ' not found.'
		except EOFError:
			print 'EOFError: not sure what it is. Path: ' + idToPath[tags[0]]
		else:
			if tags[0] not in spectrograms_file:
				S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2, n_fft = stft_window, hop_length=hop)
				#this gives us a bit over 10% compression (uncompressed size for 100 spectrograms is 372877408 bytes)
				# spectrograms_file.create_dataset(tags[0], chunks=(128,1214), shuffle=True, data=S, compression="gzip", compression_opts=9)
				#limiting precision to 10 decimal points gives ~50% compression, 8 decimals 65%
				# spectrograms_file.create_dataset(tags[0], chunks=(128,1214), shuffle=True, scaleoffset=8, data=S, compression="gzip", compression_opts=9)
				spectrograms_file.create_dataset(tags[0], chunks=(128,456), shuffle=True, scaleoffset=8, data=S, compression="gzip", compression_opts=9)
				# arr = spectrograms_file[tags[0]][()]

				#debug: compression tests
				# test+=1;
				# if test == 100:
				# 	break;

	  if index%260 == 0:
	  	print str(index*100/total_clips) + "%"
	  	# comment out to inspect the spectrogram shape
	  	# break;

#generate and display the spectrogram plot
# librosa.display.specshow(librosa.logamplitude(arr,ref_power=np.max),sr=sr, hop_length=hop, y_axis='mel', fmax=sr/2,x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
# plt.tight_layout()
# plt.show()

spectrograms_file.close()