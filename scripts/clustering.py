#!/usr/bin python

# needed imports
from matplotlib import pyplot as plt
import numpy as np
import gensim
import os
import operator

from sklearn import cluster


model_path = 'data/d2vmodel.doc2vec'
tags_file = 'data/tags'


#load model
model = gensim.models.Doc2Vec.load(model_path)

# create array of vectors
vectors = np.array([model.docvecs[clip] for clip in model.docvecs.doctags.keys()], dtype=np.float32)

# number of clusters
k = 40

kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(vectors)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

#clip_ids as numpy array
clipids = np.array(model.docvecs.doctags.keys())

#create clip_id->tags dictionary
tags_dict = dict()
with open(tags_file, "r") as f:
    for index, line in enumerate(f):
        tags = line.split()
        tags_dict[tags[0]] = tags[1:]

#create list of dictionaries where frequency of terms will be stored
#for each cluster
dictlist = [dict() for x in range(k)]

for cluster in range(k):
	print 'cluster: ' + str(cluster)
	for idx in clipids[labels==cluster]:
		for tag in tags_dict[idx[5:]]:
			if tag in dictlist[cluster]:
				dictlist[cluster][tag] += 1
    		else:
				dictlist[cluster][tag] = 1

	print sorted(dictlist[cluster].iteritems(), key=operator.itemgetter(1), reverse=True)[:5]

# centroids,_ = kmeans(smaller,10)
# # assign each sample to a cluster
# idx,_ = vq(smaller,centroids)

# print (smaller.shape)  # 150 samples with 2 dimensions

# # generate the linkage matrix
# Z = linkage(vectors, 'ward')
# # Z = linkage(smaller, 'ward')

# print(Z)

# F = fcluster(Z,1.15465)