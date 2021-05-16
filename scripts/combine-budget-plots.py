#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

import sys

# note: input files should have been generated with independent seeds

def seek_to_data(f):
    while True:
        line = f.readline()
        if not line:
            break
        if line.rstrip() == 'Scores in each bin:':
            line = f.readline()
            fields = line.split()
            assert fields[0] == 'index'
            assert fields[1] == 'ceiling'
            assert fields[2] == 'n'
            assert fields[3] == 'min'
            assert fields[4] == 'avg'
            assert fields[5] == 'max'
            break

files = []
for filename in sys.argv[1:]:
    f = open(filename, 'r')
    seek_to_data(f)
    files.append(f)

assert len(files) >= 2

f0 = files[0]
remaining_files = files[1:]
while True:
    line = f0.readline()
    if not line or line.startswith('Duration'):
        # end of data
        # --> check other files are aligned
        for f in remaining_files:
            line = f.readline()
            if not line.startswith('Duration'):
                sys.stderr.write('Training data in some files')
                break
        break
    fields  = line.split()
    index   = fields[0]
    ceiling = fields[1]
    n       = int(fields[2])
    min_score = float(fields[3])
    avg_score = n * float(fields[4])
    max_score = float(fields[5])
    total_n   = n
    for f in remaining_files:
        line = f.readline()
        fields = line.split()
        assert index   == fields[0]
        assert ceiling == fields[1]
        n = int(fields[2])
        total_n += n
        min_score = min(min_score, float(fields[3]))
        avg_score += n * float(fields[4])
        max_score = max(max_score, float(fields[5]))
    avg_score /= float(total_n)
    print('%s\t%s\t%d\t%7.2f\t%7.2f\t%7.2f' %(index, ceiling, total_n, min_score, avg_score, max_score))
