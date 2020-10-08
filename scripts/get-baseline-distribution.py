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
ls */prediction-00-[123]*-dev-*u */model-00-1-*/xx*ud-dev*u */model-00*/stdout.txt | ./get-baseline-distribution.py
or
find | ./get-baseline-distribution.py
(temporary files will be created in /dev/shm)
'''

tmp_dir = '/dev/shm'
opt_max_buckets = 16    # should be power of 2 (otherwise bucket sizes will differ hugely)
opt_combiner_repetitions = 20
opt_average = True
opt_test = False       # set to True to also create LAS distributions for test sets
opt_debug_level = 1    # 0 = quiet to 5 = all detail

opt_distribution = None
if len(sys.argv) > 1:
    if sys.argv[1] == '--distribution':
        opt_distribution = int(sys.argv[2])
        del sys.argv[1]  # delete 2 args
        del sys.argv[1]
    else:
        raise ValueError('unknown option')

opt_seed = 100
if len(sys.argv) > 1:
    if sys.argv[1] == '--seed':
        opt_seed = int(sys.argv[2])
        del sys.argv[1]  # delete 2 args
        del sys.argv[1]
    else:
        raise ValueError('unknown option')

key2filenames = {}

gold = {}
seeds = {}
have_testsets = set()

while True:
    line = sys.stdin.readline()
    if not line:
        break
    if not '/' in line:
        continue
    line = line.rstrip()
    fields = line.split('/')
    filename = fields[-1]
    folder = fields[-2]
    if filename == 'stdout.txt' and folder.startswith('model-00-'):
        if opt_debug_level > 4:
            print('Using file %s as seed file' %line)
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
        if opt_debug_level > 4:
            print('Using file %s as gold file' %line)
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
    if opt_debug_level > 4:
        print('Using file %s as prediction file' %line)
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

if opt_debug_level > 2:
    print('== Test files for each experiment ==')
    for key in sorted(list(gold.keys())):
        test_file = gold[key]
        test_tbid, test_type = key
        print('%s\t%s\t%s' %(test_tbid, test_type, test_file))

def get_split_for_buckets(candidates, n):
    if n == 1 or len(candidates) <= 1:
        retval = []
        bucket = []
        for prediction in candidates:
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
        right_half = get_split_for_buckets(candidates[split_point:], n-half_n)
        retval = left_half + right_half
    return retval

def get_tmp_name(tmp_dir, extension, prefix):
    retval = None
    while retval is None or os.path.exists(retval):
        retval = '%s/%s-%d%s' %(
            tmp_dir, prefix, random.randrange(999999), extension
        )
    return retval

def get_score(prediction_path, gold_path, tmp_dir = '/tmp'):
    eval_path = prediction_path[:-7] + '.eval.txt'
    cleanup_eval = False
    if not os.path.exists(eval_path):
        # cannot reuse existing eval.txt
        eval_path = get_tmp_name(
            tmp_dir, '.eval.txt', '%x' %abs(hash(prediction_path))
        )
        cleanup_eval = True
    if not gold_path.startswith('/') and 'UD_TREEBANK_DIR' in os.environ:
        gold_path = os.environ['UD_TREEBANK_DIR'] + '/' + gold_path
    score, score_s = dataset_module.evaluate(
        prediction_path, gold_path,
        outname = eval_path,
        reuse_eval_txt = True,
        verbose = False,
    )
    if not score:
        raise ValueError('Zero LAS for %s in %s' %(prediction_path, eval_path))
    if cleanup_eval:
        os.unlink(eval_path)
    return score

distr_counter = 0
n_distr = len(key2filenames)
for key in sorted(list(key2filenames.keys())):
    distr_counter = distr_counter + 1
    if opt_distribution and opt_distribution != distr_counter:
        continue
    print('\n\n== Distribution %d of %d: %r ==\n' %(distr_counter, n_distr, key,))
    language, parser, sample, learners, test_tbid, test_type = key
    learner2predictions = key2filenames[key]
    assert len(learner2predictions) == learners
    learner_buckets = []
    learner_keys = sorted(list(learner2predictions.keys()))
    for learner in learner_keys:
        print('\n=== Learner %s ===\n' %learner)
        predictions = learner2predictions[learner]
        predictions.sort()
        n = len(predictions)
        if opt_debug_level > 2:
            print('number of predictions: %d' %n)
            for i in range(n):
                print('[%d] = %r' %(i, predictions[i]))
        # predictions[i] = (exp_code, path)
        if n < opt_max_buckets:
            sys.stderr.write('Warning: only %d predictions for learner %s and %r\n' %(
                n, learner, key
            ))
        buckets = []
        if True:
            # get LAS for each prediction
            candidates = []
            for prediction in predictions:
                exp_code, path = prediction
                score = get_score(path, gold[(test_tbid, test_type)], tmp_dir = tmp_dir)
                seed  = seeds[(exp_code, learner)]
                candidates.append((score, seed, exp_code, path))
            candidates.sort()
            # find best split: must have sufficient items on each side and
            # minimise seed overlap (break ties by preferring balanced splits)
            buckets = get_split_for_buckets(candidates, opt_max_buckets)
        if opt_debug_level > 0:
            print('Buckets:')
            j = 0
            for i, bucket in enumerate(buckets):
                print('%d:' %i)
                for item in bucket:
                    assert item == candidates[j]
                    print('\t%.3f %9s %s %s' %(candidates[j]))
                    j = j + 1
        learner_buckets.append(buckets)
    # enumerate all combinations of buckets, one from each learner
    print('=== Bucket combinations ===')
    distr_scores = []
    bucket_comb_index = 0
    n_bucket_combinations = 1
    for buckets in learner_buckets:
        n_bucket_combinations *= len(buckets)
    if opt_debug_level > 0:
        print('number of combinations:', n_bucket_combinations)
    for bucket_combination in apply(itertools.product, learner_buckets):
        start_time = time.time()
        print('[%d]:' %bucket_comb_index)
        # pick a prediction from each bucket
        predictions = []
        filenames = []
        for bucket in bucket_combination:
            choice = random.choice(bucket)
            predictions.append(choice)
            filenames.append(choice[3])
            print('\t', choice)
        scores = []
        next_comb_seed = opt_seed*n_distr
        next_comb_seed += distr_counter-1
        next_comb_seed *= n_bucket_combinations
        next_comb_seed += bucket_comb_index
        next_comb_seed *= opt_combiner_repetitions
        for j in range(opt_combiner_repetitions):
            output_path = get_tmp_name(
                tmp_dir, '.conllu',
                '%s-%s-%s-%d-%s-%s-%d-%d' %(
                    language, parser, sample, learners, test_tbid, test_type,
                    bucket_comb_index, j,
            ))
            dataset_module.combine(
                filenames, output_path,
                seed = '%d' %next_comb_seed,
                verbose = False
            )
            scores.append(get_score(
                 output_path, gold[(test_tbid, test_type)],
                 tmp_dir = tmp_dir
            ))
            os.unlink(output_path)
            next_comb_seed = next_comb_seed + 1
        scores.sort()
        average_score = sum(scores) / float(len(scores))
        print('\tscores = %r' %(scores,))
        print('\tduration = %.1fs' %(time.time()-start_time))
        print('\taverage score = %.9f (%.2f)' %(average_score, average_score))
        print('\tmax-min = %.9f (%.2f)' %(max(scores)-min(scores), max(scores)-min(scores)))
        sq_errors = []
        for score in scores:
            error = average_score - score
            sq_errors.append(error**2)
        n = len(scores)
        std_dev = (sum(sq_errors)/float(n))**0.5
        print('\tpopulation std dev = %.9f (%.2f)' %(std_dev, std_dev))
        std_dev = (sum(sq_errors)/(n-1.0))**0.5
        print('\tsimple sample std dev = %.9f (%.2f)' %(std_dev, std_dev))
        std_dev = (sum(sq_errors)/(n-1.5))**0.5
        print('\tapproximate std dev = %.9f (%.2f)' %(std_dev, std_dev))
        std_dev = (sum(sq_errors)/(n-1.5+1.0/(8.0*(n-1.0))))**0.5
        print('\tmore accurate std dev = %.9f (%.2f)' %(std_dev, std_dev))
        if opt_average:
            scores = []
            scores.append(average_score)
        for s_index, score in enumerate(scores):
            info = []
            info.append('%03x' %bucket_comb_index)
            if not opt_average:
                info.append('%d' %s_index)
            for learner_score, _, _, _ in predictions:
                info.append('%.2f' %learner_score)
            info.append('%.3f' %std_dev)
            info.append('%.2f' %min(scores))
            info.append('%.2f' %max(scores))
            for _, seed, _, _ in predictions:
                info.append(seed)
            for _, _, _, path in predictions:
                info.append(path)
            distr_scores.append((score, '\t'.join(info)))
        bucket_comb_index = bucket_comb_index + 1
        if opt_debug_level > 3:
            distr_scores.sort()
            print('\tdistribution so far:')
            for score, info in distr_scores:
                print('\t%.9f\t%s' %(score, info))
        sys.stdout.flush()
    distr_scores.sort()
    if sample == '-':
        sample = 's'
    d_name = 'distribution-%s%s%s-%d-%s-%s.txt' %(
        language, parser, sample, learners, test_tbid, test_type,
    )
    f = open(d_name, 'wb')
    for score, info in distr_scores:
        f.write('%.9f\t%s\n' %(score, info))
    f.close()

if opt_debug_level > 4:
    print('\n\n== Seeds ==\n')
    for key in sorted(list(seeds.keys())):
        exp_code, learner = key
        seed = seeds[key]
        print('%s\t%s\t%s' %(exp_code, learner, seed))

