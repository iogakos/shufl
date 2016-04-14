# credits for this to http://rare-technologies.com/doc2vec-tutorial/
class MagnaTags(object):
  def __init__(self, filename):
    self.filename = filename

  def __iter__(self):
    for line in open(self.filename):
      tags = line.split()
      yield gensim.models.doc2vec.LabeledSentence(words=tags[1:], tags=['CLIP_%s' % tags[0]])

import gensim
import os

model_path = 'data/d2vmodel.doc2vec'

# uncomment for running on aws
data_root = '/mnt/'
model_path = data_root + model_path

try:
    model = gensim.models.Doc2Vec.load(model_path)
except IOError:
	tags = MagnaTags('data/tags') # a memory-friendly iterator
	#bigram_transformer = gensim.models.Phrases(tags)
	#model = gensim.models.Word2Vec(bigram_transformer[tags], size=100, min_count = 1)
	model = gensim.models.Doc2Vec(alpha=0.025, min_alpha=0.025, size = 40)
	model.build_vocab(tags)
	for epoch in range(10):
	  model.train(tags)
	  model.alpha -= 0.002  # decrease the learning rate
	  model.min_alpha = model.alpha  # fix the learning rate, no decay

	#save the model
	model.save(model_path)
