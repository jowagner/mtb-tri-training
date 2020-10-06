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
import random
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

class LazyReadSeeds:
    def __init__(self, d):
        self.d = d
    def __getitem__(self, key):
        seed = self.d[key]
        if seed.endswith('.txt'):
            seed = get_seed(seed)
            self.d[key] = seed
        return seed
    def keys(self):
        return self.d.keys()

seeds = LazyReadSeeds(seeds)

if opt_debug:
    print('== Test files for each experiment ==')
    for key in sorted(list(gold.keys())):
        test_file = gold[key]
        test_tbid, test_type = key
        print('%s\t%s\t%s' %(test_tbid, test_type, test_file))

def get_split_for_buckets(candidates, n):
    # n must be power of two
    assert bin(n).count('1') == 1
    if n == 1:
        retval = []
        bucket = []
        for _, _, prediction in candidates:
            bucket.append(prediction)
        retval.append(bucket)
    else:
        # find best split: must have sufficient items on each side and
        # minimise seed overlap (break ties by preferring balanced splits)
        half_n = int(n / 2)
        candidate_splits = []
        for split_point in range(1, len(candidates)):
            size_1 = split_point
            size_2 = len(candidates) - size_1
            if min(size_1, size_2) < half_n:
                # cannot split this way as one half would be too small
                continue
            balance = abs(size_1 - size_2)
            # check seed overlap
            left_seeds = set()
            right_seeds = set()
            for i in range(len(candidates)):
                if i < size_1:
                    left_seeds.add(candidates[i][1])
                else:
                    right_seeds.add(candidates[i][1])
            intersection = left_seeds & right_seeds
            seed_overlap = len(intersection)
            candidate_splits.append((seed_overlap, balance, split_point, intersection, size_1, size_2))
        candidate_splits.sort()
        # get best split
        seed_overlap, balance, split_point, intersection, size_1, size_2 = candidate_splits[0]
        if seed_overlap or balance > 1:
            sys.stderr.write('Cannot avoid seed overlap or imbalance:\n')
            sys.stderr.write('\thalf_n = %d\n' %half_n)
            sys.stderr.write('\tsize_1 = %d\n' %size_1)
            sys.stderr.write('\tsize_2 = %d\n' %size_2)
            sys.stderr.write('\toverlap = %d, intersection = %r\n' %(seed_overlap, intersection))
            for i, candidate in enumerate(candidates):
                 sys.stderr.write('\t[%d] = %r\n' %(i, candidate))
        left_half  = get_split_for_buckets(candidates[:split_point], half_n)
        right_half = get_split_for_buckets(candidates[split_point:], half_n)
        retval = left_half + right_half
    return retval

def get_score(prediction_path, gold_path, tmp_dir = '/tmp'):
    eval_path = prediction_path[:-7] + '.eval.txt'
    cleanup_eval = False
    if not os.path.exists(eval_path):
        # cannot reuse existing eval.txt
        eval_path = tmp_dir + '/0.eval.txt'   # TODO: make more robust for parallel runs
        cleanup_eval = True
        while os.path.exists(eval_path):
            eval_path = '%s/%d.eval.txt' %(tmp_dir, random.randrange(99999))
    if not gold_path.startswith('/') and 'UD_TREEBANK_DIR' in os.environ:
        gold_path = os.environ['UD_TREEBANK_DIR'] + '/' + gold_path
    score, score_s = dataset_module.evaluate(
        prediction_path, gold_path,
        outname = eval_path,
        reuse_eval_txt = True
    )
    if not score:
        raise ValueError('Zero LAS for %s in %s' %(prediction_path, eval_path))
    if cleanup_eval:
        os.unlink(eval_path)
    return score

for key in sorted(list(key2filenames.keys())):
    print('\n\n== Distribution for %r ==\n' %(key,))
    language, parser, sample, learners, test_tbid, test_type = key
    learner2predictions = key2filenames[key]
    assert len(learner2predictions) == learners
    learner2buckets = {}
    learner_keys = sorted(list(learner2predictions.keys()))
    for learner in learner_keys:
        print('\n=== Learner %s ===\n' %learner)
        predictions = learner2predictions[learner]
        predictions.sort()
        n = len(predictions)
        if opt_debug:
            print('number of predictions: %d' %n)
            for i in range(n):
                print('[%d] = %r' %(i, predictions[i]))
        # predictions[i] = (exp_code, path)
        if n < 16:
            sys.stderr.write('Warning: only %d predictions for learner %s and %r\n' %(
                n, learner, key
            ))
        buckets = []
        if n <= 16:
            # each bucket gets exactly one prediction
            for prediction in predictions:
                bucket = []
                bucket.append(prediction)
                buckets.append(bucket)
        else:
            # get LAS for each prediction
            candidates = []
            for prediction in predictions:
                exp_code, path = prediction
                score = get_score(path, gold[(test_tbid, test_type)], tmp_dir = tmp_dir)
                seed  = seeds[(exp_code, learner)]
                candidates.append((score, seed, prediction))
            candidates.sort()
            # find best split: must have sufficient items on each side and
            # minimise seed overlap (break ties by preferring balanced splits)
            buckets = get_split_for_buckets(candidates, 16)
        if opt_debug:
            print('Buckets:')
            j = 0
            for i, bucket in enumerate(buckets):
                print('%d:' %i)
                for item in bucket:
                    assert item == candidates[j][2]
                    print('\t%.3f %9s %r' %(candidates[j]))
                    j = j + 1
        learner2buckets[learner] = buckets
    break
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

if opt_debug:
    print('== Seeds ==')
    for key in sorted(list(seeds.keys())):
        exp_code, learner = key
        seed = seeds[key]
        print('%s\t%s\t%s' %(exp_code, learner, seed))

