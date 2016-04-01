# as taken from http://rare-technologies.com/word2vec-tutorial/
class MagnaTags(object):
  def __init__(self, dirname):
    self.dirname = os.path.abspath(dirname)

  def __iter__(self):
    for fname in os.listdir(self.dirname):
      for line in open(os.path.join(self.dirname, fname)):
        yield line.split()

import gensim
import os

tags = MagnaTags('data') # a memory-friendly iterator
#bigram_transformer = gensim.models.Phrases(tags)
#model = gensim.models.Word2Vec(bigram_transformer[tags], size=100, min_count = 1)
model = gensim.models.Word2Vec(tags, size=100, min_count=1)
