#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

from __future__ import print_function

import conllu_dataset as dataset_module

import itertools
import os
import sys

'''
usage:
ls */prediction-00-?-*-dev-*.conllu */model-00-1-*/*.conllu | ./get-baseline-distribution.py
(temporary files will be created in /dev/shm)
'''

tmp_dir = '/dev/shm'

key2filenames = {}

exp2gold = {}

while True:
    line = sys.stdin.readline()
    if not line:
        break
    line = line.rstrip()
    fields = line.split('/')
    exp_code = fields[-2]
    if exp_code.startswith('model-00-1'):
        exp_code = fields[-3]
        is_modelfile = True
        if 'train' in fields[-1]:
            continue
        target = os.readlink(line)
        test_type = target.replace('.', '-').split('-')[-2]
        exp2gold[(exp_code, test_type)] = '/'.join(target.split('/')[-2:])
        continue
    language = exp_code[0]
    parser   = exp_code[1]
    sample   = exp_code[3]
    learners = int(exp_code[-2])
    fields = fields[-1].split('-')
    learner = fields[2]
    if learner == 'E':
        continue
    seed = exp_code[-2:] + learner
    test_tbid = fields[3]
    test_type = fields[4]
    key = (language, parser, sample, learners, test_tbid, test_type)
    if not key in key2filenames:
        key2filenames[key] = []
    key2filenames[key].append((seed, exp_code, line))

for exp_code, test_type in exp2gold:
    print('%s\t%s\t%s' %(exp_code, test_type, exp2gold[(exp_code, test_type)]))

for key in key2filenames:
    print('Combinations for', key)
    seeds_and_filenames = key2filenames[key]
    n = len(seeds_and_filenames)
    total = 0
    for _ in itertools.combinations(range(n), key[3]):
        total += 1
    scores = []
    count = 0
    for indices in itertools.combinations(range(n), key[3]):
        count += 1
        print('Combinations %d of %d for %r' %(count, total, key))
        seeds = set()
        filenames = []
        s_indices = []
        gold_path = None
        seed_conflict = False
        for index in indices:
            seed, exp_code, filename = seeds_and_filenames[index]
            if seed in seeds:
                seed_conflict = True
                break
            seeds.add(seed)
            filenames.append(filename)
            s_indices.append('%d' %index)
            candidate_path = exp2gold[(exp_code, test_type)]
            if gold_path is None:
                gold_path = candidate_path
            elif gold_path != candidate_path:
                raise ValueError('gold path inconsistency')
        if seed_conflict:
            continue
        s_indices = '-'.join(s_indices)
        d_key = (tmp_dir,) + key + (s_indices,)
        output_path = '%s/distribution-%s%s%s-%d-%s-%s-%s.conllu' %d_key
        dataset_module.combine(filenames, output_path)
        score, score_s = dataset_module.evaluate(
            output_path, os.environ['UD_TREEBANK_DIR'] + '/' + gold_path
        )
        scores.append((score, score_s, ) + tuple(filenames))
        os.unlink(output_path)
        os.unlink(output_path[:-7] + '.eval.txt')
    scores.sort()
    f = open('distribution-%s%s%s-%d-%s-%s.txt' %key, 'wb')
    for score in scores:
        score = score[1:]
        f.write('\t'.join(score))
        f.write('\n')
    f.close()

