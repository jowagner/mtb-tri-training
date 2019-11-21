#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

import os
import sys
import time

"""
Usage: grep -HE "(Score|Iteration)" */stdout.txt | ./log-to-scores.py > scores.tsv
or   : grep -HE "(Score|Iteration)" */stdout.txt | ./log-to-scores.py --update scores.tsv
"""

key2scores = {}
last_filename = None
max_rounds = 0
while True:
    line = sys.stdin.readline()
    if not line:
        break
    fields = line.replace(':', ' ').split()
    filename = fields[0]
    if filename != last_filename:
        code = filename.split('/')[-2]
        lang = code[0]
        parser = code[1]
        method = code[2:6]
        num_learners = int(code[6])
        augment_size_index = int(code[7])
        last_filename = filename
        tt_round = 0
        score_index = 0
        test_set_index = 0
    if '== Tri-training Iteration' in line:
        tt_round += 1
        if fields[-4] != '%d' %tt_round:
            raise ValueError('logfile %s contains data for round %s where data for round %d is expected' %(
                filename, fields[-4], tt_round
            ))
        score_index = 0
        test_set_index = 0
    else:
        if tt_round > max_rounds:
            max_rounds = tt_round
        score = fields[-1]
        if score_index < num_learners:
            learner = 'L%d' %(score_index+1)
        else:
            learner = 'Ensemble'
        key = (
            lang, parser,
            '%d' %augment_size_index,
            method, '%d' %num_learners,
            learner, '%d' %test_set_index
        )
        if not key in key2scores:
            key2scores[key] = []
        key2scores[key].append(score)
        if score_index == num_learners:
            score_index = 0
            test_set_index += 1
        else:
            score_index += 1

key_and_rounds_header = '\t'.join([
    'Language', 'Parser',
    'AugmentSizeIndex',
    'Method', 'NumberOfLearners',
    'Learner', 'TestSetIndex', 'Rounds'
])

backup_stdout = sys.stdout
if len(sys.argv) > 1:
    if sys.argv[1] != '--update':
        raise ValueError('unknown option %r' %(sys.argv[1]))
    filename = sys.argv[2]
    f = open(filename, 'rb')
    old_header = f.readline()
    if not old_header.startswith(key_and_rounds_header):
        raise ValueError('Unsupported tsv format')
    while True:
        line = f.readline()
        if not line:
            break
        fields = line.split()
        key = tuple(fields[:7])
        old_scores = fields[8:]
        if key not in key2scores \
        or len(scores) > len(key2scores[key]):
            key2scores[key] = scores
    f.close()
    sys.stdout = open(filename, 'wb')

# table header
sys.stdout.write(key_and_rounds_header)
for i in range(max_rounds+1):
    sys.stdout.write('\t%d' %i)
sys.stdout.write('\n')
# table body
for key in sorted(key2scores):
    sys.stdout.write('\t'.join(key))
    scores = key2scores[key]
    sys.stdout.write('\t%d\t' %(len(scores)-1))
    sys.stdout.write('\t'.join(scores))
    sys.stdout.write('\n')

if sys.stdout != backup_stdout:
    sys.stdout.close()
    sys.stdout = backup_stdout

