#!/usr/bin/env python

import sys

opt_skip_minmax = True

header = []
header.append('AvgEnsLAS')
header.append('Buckets')
header.append('LAS-1')
header.append('LAS-2')
header.append('LAS-3')
header.append('StddevEnsLAS')
if not opt_skip_minmax:
    header.append('MinEnsLAS')
    header.append('MaxEnsLAS')
header.append('Seed-1')
header.append('Seed-2')
header.append('Seed-3')
header.append('Duration-1')
header.append('Duration-2')
header.append('Duration-3')
header.append('Pred-1')
header.append('Pred-2')
header.append('Pred-3')

sys.stdout.write('\t'.join(header))
sys.stdout.write('\n')

while True:
    line = sys.stdin.readline()
    if not line:
        break
    fields = line.split()
    if opt_skip_minmax:
        del fields[7]
        del fields[6]
    fields[1] = "'" + fields[1]
    sys.stdout.write('\t'.join(fields))
    sys.stdout.write('\n')
    
