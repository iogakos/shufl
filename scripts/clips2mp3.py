#!/bin/python

import sys

for arg in sys.argv[1:]:
    with open('data/idToPath') as f:
        for line in f:
            clip_id, filename = line[:-1].split()            
            if arg == clip_id or arg[5:]== clip_id:
                print 'data/magna/' + filename
                break
