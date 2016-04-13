#!/bin/python

import csv
import re

f = open('data/annotations_final.csv', 'rb')
reader = csv.reader(f, delimiter='\t')
tags_list = list(reader)
tags = [re.sub("\s", "_", t) for t in tags_list[0]]

tags_file = open('data/tags', 'w');
idToPath_file = open('data/idToPath', 'w');

for row in tags_list[1:]:
  tmp = []
  for idx, count in enumerate(row[:-1]):
    if count > '0':
      tmp.append(row[idx] if idx == 0 else tags[idx])
  if(len(tmp) > 1):
    tags_file.write("%s\n" % ' '.join(tmp))

  idToPath_file.write(row[0] + ' '+ row[-1] + '\n')
