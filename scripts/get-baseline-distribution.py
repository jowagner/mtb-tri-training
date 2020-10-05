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
import time

'''
usage:
ls */prediction-00-?-*-dev-*.conllu */model-00-1-*/*.conllu */model-00*/cmd | ./get-baseline-distribution.py
or
find | ./get-baseline-distribution.py
(temporary files will be created in /dev/shm)
'''

tmp_dir = '/dev/shm'
opt_test = False       # set to True to also create LAS distributions for test sets
opt_debug = True

key2filenames = {}

gold = {}
seeds = {}
have_testsets = set()

while True:
    line = sys.stdin.readline()
    if not line:
        break
    line = line.rstrip()
    fields = line.split('/')
    filename = fields[-1]
    folder = fields[-2]
    if filename == 'stdout.txt' and folder.startswith('model-00-'):
        exp_code = fields[-3]
        learner = folder.split('-')[2]
        seeds[(exp_code, learner)] = line
        continue
    if not filename.endswith('.conllu'):
        continue
    if folder.startswith('model-00-1') \
    and not filename.endswith('-ud-train.conllu'):
        # note that the test type in `filename` can be wrong
        # due to an earlier bug in the training wrapper script
        exp_code = fields[-3]
        language = exp_code[0]
        have_all_test_types = True
        for test_type in ('dev', 'test'):
            if test_type == 'test' and not opt_test:
                continue
            if not (language, test_type) in have_testsets:
                have_all_test_types = False
                break
        if have_all_test_types:
            continue
        target = os.readlink(line)
        t_filename = target.split('/')[-1]
        fields = t_filename.replace('.', '-').split('-')
        test_tbid = fields[0]
        if fields[1] != 'ud' \
        or fields[3] != 'conllu':
            sys.stderr.write('Unexpected file %r\n' %line)
            continue
        test_type = fields[2]
        if test_type == 'test' and not opt_test:
            continue
        gold[(test_tbid, test_type)] = '/'.join(target.split('/')[-2:])
        have_testsets.add((language, test_type))
        continue
    elif folder.startswith('model-') \
    and folder[6].isdigit() \
    and folder[7].isdigit() \
    and folder[8] == '-' \
    and folder[9].isdigit():
        continue
    elif len(folder) != 8:
        sys.stderr.write('Ignoring unexpected file %r\n' %line)
        continue
    exp_code = folder
    language = exp_code[0]
    parser   = exp_code[1]
    sample   = exp_code[3]
    try:
        learners = int(exp_code[-2])
    except:
        raise ValueError('Unsupported exp_code %r in %s' %(exp_code, line))
    run = 1
    if learners == 4:
        learners = 3
        run = 2
    fields = fields[-1].split('-')
    if fields[0] != 'prediction':
        continue
    if len(fields) < 5:
        sys.stderr.write('Not enough fields in filename %r\n' %line)
        continue
    if fields[1] != '00':
        continue
    learner = fields[2]
    if learner == 'E':
        continue
    test_tbid = fields[3]
    test_type = fields[4]
    if test_type != 'dev' and not opt_test:
        continue
    key = (language, parser, sample, learners, test_tbid, test_type)
    if not key in key2filenames:
        key2filenames[key] = {}
    learner2filenames = key2filenames[key]
    if not learner in learner2filenames:
        learner2filenames[learner] = []
    learner2filenames[learner].append((exp_code, line))

def get_seed(filename):
    retval = 'unknown'
    f = open(filename, 'rb')
    while True:
        line = f.readline()
        if not line:
            break
        if line.startswith('seed:'):
            retval = line.split()[1]
            break
    f.close()
    return retval

n_seeds = len(seeds)
i = 0
start = time.time()
for key in list(seeds.keys())[:]:
    i = i + 1
    sys.stderr.write('Reading seed %d of %d (%.1f%% done)...\r' %(i, n_seeds, 100.0*(i-1)/float(n_seeds)))
    filename = seeds[key]
    seeds[key] = get_seed(filename)
duration = time.time() - start
sys.stderr.write('Read %s seeds in %.1f seconds     \n' %(n_seeds, duration))

if opt_debug:
    print('== Test files for each experiment ==')
    for key in sorted(list(gold.keys())):
        test_file = gold[key]
        test_tbid, test_type = key
        print('%s\t%s\t%s' %(test_tbid, test_type, test_file))
    print('== Seeds ==')
    for key in sorted(list(seeds.keys())):
        exp_code, learner = key
        seed = seeds[key]
        print('%s\t%s\t%s' %(exp_code, learner, seed))

for key in sorted(list(key2filenames.keys())):
    print('== %r ==' %(key,))
    language, parser, sample, learners, test_tbid, test_type = key
    learner2filenames = key2filenames[key]
    assert len(learner2filenames) == learners
    learner_partitions = []
    for learner in sorted(list(learner2filenames.keys())):
        print('=== Learner %s ===' %learner)
        filenames = learner2filenames[learner]
        filenames.sort()
        n = len(filenames)
        if opt_debug:
            print('number of predictions: %d' %n)
            for i in range(n):
                print('[%d] = %r' %(i, filenames[i]))
        # filenames[i] = (exp_code, path)
        if n < 16:
            sys.stderr.write('Warning: only %d predictions for learner %s and %r\n' %(
                n, learner, key
            ))
        partitions = []
        if n <= 16:
            for prediction in filenames:
                partition = []
                partition.append(prediction)
                partitions.append(partition)
        else:
            # TODO:
            # get LAS for each prediction
            # find best split: must have sufficient items on each side and
            # minimise seed overlap (break ties by preferring balanced splits)
            raise NotImplementedError
        learner_partitions.append(partitions)
    continue
    # TODO: adjust below to use partitions
    total = 0
    for _ in itertools.combinations(range(n), learners):
        total += 1
    scores = []
    count = 0
    for indices in itertools.combinations(range(n), learners):
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
    d_name = 'distribution-%s%s%s-%d-%s-%s.txt' %key
    d_name = d_name.replace('--', 's-')
    f = open(d_name, 'wb')
    for score in scores:
        score = score[1:]
        f.write('\t'.join(score))
        f.write('\n')
    f.close()

