#!/bin/python

import csv
f = open('data/annotations_final.csv', 'rb')
reader = csv.reader(f, delimiter='\t')
tags_list = list(reader)
tags = tags_list[0]

tags_file = open('data/tags', 'w');

for row in tags_list[1:]:
  tmp = []
  for idx, count in enumerate(row[1:-1]):
    if count > '0':
      tmp.append(tags[idx + 1])
  if(len(tmp) > 0):
    tags_file.write("%s\n" % ' '.join(tmp))
