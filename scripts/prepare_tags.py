#!/bin/python

import csv
f = open('data/annotations_final.csv', 'rb')
reader = csv.reader(f, delimiter='\t')
tags_list = list(reader)
tags = tags_list[0]

tags_file = open('data/tags', 'w');

# TODO: we are not interested in the samples with no information about the tags
# dont put it in the list, or leavet it to higher level script to decide
# TODO: might need to convert negating tags to not contain spaces
# i.e 'no vocals' to 'no_vocals'
for row in tags_list[1:]:
  tmp = []
  for idx, count in enumerate(row[:-1]):
    if count > '0':
      tmp.append(row[idx] if idx == 0 else tags[idx])
  if(len(tmp) > 0):
    tags_file.write("%s\n" % ' '.join(tmp))
