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
ls */prediction-00-[123]*-dev-*.conllu* */model-00-1-*/xx*ud-dev*u */model-00*/stdout.txt | ./get-baseline-distribution.py
or
find | ./get-baseline-distribution.py
(temporary files will be created in /dev/shm)
'''

tmp_dir = '/dev/shm'
opt_max_buckets = 16    # should be power of 2 (otherwise bucket sizes will differ hugely)
opt_combiner_repetitions = 21
opt_average = True
opt_test = False       # set to True to also create LAS distributions for test sets
opt_debug_level = 2    # 0 = quiet to 5 = all detail
opt_distribution = None
opt_dry_run = False    # only list distribitions, do not run the combiner or evaluation
opt_partial_distribution = None    # set to 1 to opt_parts (inclusive) to select a part
opt_parts = 9
opt_seed = 100
opt_languages = None  # no restriction on the language
opt_parsers   = None  # no restriction on the parser

while len(sys.argv) > 1 and sys.argv[1].startswith('--'):
    option = sys.argv[1]
    del sys.argv[1]
    if option == '--distribution':
        opt_distribution = int(sys.argv[1])
        if opt_distribution < 0:
            # a way to keep this deactivated while using the option
            opt_distribution = None
        del sys.argv[1]
    elif option == '--part':
        opt_partial_distribution = int(sys.argv[1])
        assert opt_partial_distribution != 0  # 1 = first part
        if opt_partial_distribution < 0:
            # a way to keep parts deactivated while using --part
            opt_partial_distribution = None
        del sys.argv[1]
    elif option == '--parts':
        opt_parts = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--seed':
        opt_seed = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--tmp-dir':
        tmp_dir = sys.argv[1]
        del sys.argv[1]
    elif option == '--buckets':
        opt_max_buckets = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--repetitions':
        opt_combiner_repetitions = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--individual-scores':
        opt_average = False
    elif option == '--include-test':
        opt_test = True
    elif option == '--languages':
        opt_languages = sys.argv[1]
        del sys.argv[1]
    elif option == '--parsers':
        opt_parsers = sys.argv[1]
        del sys.argv[1]
    elif option in ('--dry-run', '--list-distributions'):
        opt_dry_run = True
    elif option == '--quiet':
        opt_debug_level = 0
    elif option == '--verbose':
        opt_debug_level = 4
    elif option == '--debug':
        opt_debug_level = 5
    else:
        raise ValueError('unknown option')

if opt_debug_level >= 5:
    print('tmp_dir:', tmp_dir)
    print('opt_max_buckets:', opt_max_buckets)
    print('opt_combiner_repetitions:', opt_combiner_repetitions)
    print('opt_average:', opt_average)
    print('opt_test:', opt_test)
    print('opt_debug_level:', opt_debug_level)
    print('opt_distribution:', opt_distribution)
    print('opt_dry_run:', opt_dry_run)
    print('opt_partial_distribution:', opt_partial_distribution)
    print('opt_parts:', opt_parts)
    print('opt_seed:', opt_seed)
    print('opt_languages:', opt_languages)
    print('opt_parsers:', opt_parsers)


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
    if folder.startswith('model-') and '-incomplete-' in folder:
        continue
    if folder.endswith('-workdir'):
        continue
    if filename == 'stdout.txt' and folder.startswith('model-00-'):
        if opt_debug_level > 4:
            print('Using file %s as seed file' %line)
        exp_code = fields[-3]
        learner = folder.split('-')[2]
        seeds[(exp_code, learner)] = line
        continue
    if not (filename.endswith('.conllu') or filename.endswith('.conllu.bz2')):
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
    if opt_languages and language not in opt_languages:
        if opt_debug_level > 4:
            print('Ignoring language %s (line %s)' %(language, line))
        continue
    if opt_parsers   and parser   not in opt_parsers:
        if opt_debug_level > 4:
            print('Ignoring parser %s (line %s)' %(parser, line))
        continue
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
        self.d = {}
        for k in d:
            self.d[k] = d[k]
    def __getitem__(self, key):
        seed = self.d[key]
        if seed.endswith('.txt'):
            seed = get_seed(seed)
            self.d[key] = seed
        return seed
    def keys(self):
        return self.d.keys()


def get_duration(filename):
    assert filename.endswith('stdout.txt')
    folder = filename[:-10]
    try:
        start_t = os.path.getmtime(folder + 'training.start')
        #start_s = '%.1f' %start_t
    except:
        start_t = None
        #start_s = 'unknown'
    try:
        end_t = os.path.getmtime(folder + 'training.end')
        #end_s = '%.1f' %end_t
    except:
        end_t = None
        #end_s = 'unknown'
    if start_t and end_t:
        duration = '%.3f' %(end_t - start_t)
    else:
        duration = 'unknown'
    #return '%s %s %s' %(start_s, end_s, duration)
    return duration

class LazyReadDurations:
    def __init__(self, d):
        self.d = {}
        for k in d:
            self.d[k] = d[k]
    def __getitem__(self, key):
        duration = self.d[key]
        if duration.endswith('.txt'):
            duration = get_duration(duration)
            self.d[key] = duration
        return duration
    def keys(self):
        return self.d.keys()

tr_durations = LazyReadDurations(seeds)
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
                # avoid splitting this way as one half would be too small
                priority = 5000
            else:
                priority = 0
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
            priority += 29 * seed_overlap + 11 * balance
            candidate_splits.append((
                priority, seed_overlap, balance,
                split_point, intersection, size_1, size_2
            ))
        candidate_splits.sort()
        # get best split
        _, seed_overlap, balance, split_point, intersection, size_1, size_2 = candidate_splits[0]
        if (seed_overlap or balance > 1) and opt_debug_level > 1:
            print('Cannot avoid seed overlap or imbalance:')
            print('\thalf_n = %d' %half_n)
            print('\tsize_1 = %d' %size_1)
            print('\tsize_2 = %d' %size_2)
            print('\toverlap = %d, intersection = %r' %(seed_overlap, intersection))
            for i, candidate in enumerate(candidates):
                 print('\t[%d] = %r' %(i, candidate))
        left_half  = get_split_for_buckets(candidates[:split_point], half_n)
        right_half = get_split_for_buckets(candidates[split_point:], n-half_n)
        retval = left_half + right_half
    return retval

def get_tmp_name(tmp_dir, extension, prefix):
    #if not os.path.exists(tmp_dir):
    #    raise ValueError('tmp_dir %r disappeared unexpectedly in get_tmp_name()' %tmp_dir)
    retval = None
    while retval is None or os.path.exists(retval):
        retval = '%s/%s-%x%s' %(
            tmp_dir, prefix, random.getrandbits(28), extension
        )
    #print('\t\ttmp_name %s' %retval)
    return retval

def get_score(prediction_path, gold_path, tmp_dir = '/tmp', tmp_prefix = 'u'):
    eval_path = prediction_path[:-7] + '.eval.txt'
    cleanup_eval = False
    if not os.path.exists(eval_path):
        # cannot reuse existing eval.txt
        eval_path = get_tmp_name(
            tmp_dir, '.eval.txt', '%s-%x' %(tmp_prefix, abs(hash(prediction_path)))
        )
        cleanup_eval = True
    if not gold_path.startswith('/') and 'UD_TREEBANK_DIR' in os.environ:
        gold_path = os.environ['UD_TREEBANK_DIR'] + '/' + gold_path
    #if not os.path.exists(tmp_dir):
    #    raise ValueError('tmp_dir %r disappeared unexpectedly in get_score()' %tmp_dir)
    score, score_s, eval_path = dataset_module.evaluate(
        prediction_path, gold_path,
        outname = eval_path,
        reuse_eval_txt = True,
        verbose = False,
    )
    if not score:
        raise ValueError('Zero LAS for %s in %s' %(prediction_path, eval_path))
    if cleanup_eval and eval_path:
        os.unlink(eval_path)
        #print('\t\tunlinked %s' %eval_path)
    return score

distr_counter = 0
n_distr = len(key2filenames)
if opt_debug_level >= 5:
    print('Found', n_distr, 'distribution(s)')

for key in sorted(list(key2filenames.keys())):
    distr_counter = distr_counter + 1
    if opt_distribution and opt_distribution != distr_counter:
        continue
    print('\n\n== Distribution %d of %d: %r ==\n' %(distr_counter, n_distr, key,))
    if opt_dry_run:
        continue
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
                score = get_score(
                    path, gold[(test_tbid, test_type)],
                    tmp_dir = tmp_dir,
                    tmp_prefix = '%d-%s' %(
                       distr_counter, learner,
                ))
                seed  = seeds[(exp_code, learner)]
                tr_duration = tr_durations[(exp_code, learner)]
                candidates.append((score, seed, exp_code, path, tr_duration))
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
                    print('\t%.3f %9s %s %s %s' %(candidates[j]))
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
        if opt_partial_distribution \
        and (opt_partial_distribution + bucket_comb_index) % opt_parts != 0:
            bucket_comb_index = bucket_comb_index + 1
            continue
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
                '%s-%s-%s-%d-%s-%s-%d-%d-%d' %(
                    language, parser, sample, learners, test_tbid, test_type,
                    distr_counter,
                    bucket_comb_index, j,
            ))
            #print('\t\tcombining to %s' %output_path)
            dataset_module.combine(
                filenames, output_path,
                seed = '%d' %next_comb_seed,
                verbose = False
            )
            scores.append(get_score(
                 output_path, gold[(test_tbid, test_type)],
                 tmp_dir = tmp_dir,
                 tmp_prefix = '%d-%d-%d' %(
                    distr_counter, bucket_comb_index, j
            )))
            os.unlink(output_path)
            #print('\t\tunlinked %s' %output_path)
            next_comb_seed = next_comb_seed + 1
        scores.sort()
        average_score = sum(scores) / float(len(scores))
        print('\tscores = %r' %(scores,))
        print('\tduration = %.1fs' %(time.time()-start_time))
        print('\taverage score = %.9f (%.2f)' %(average_score, average_score))
        max_score = max(scores)
        min_score = min(scores)
        print('\tmax-min = %.9f (%.2f)' %(max_score-min_score, max_score-min_score))
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
            for learner_score, _, _, _, _ in predictions:
                info.append('%.2f' %learner_score)
            info.append('%.3f' %std_dev)
            info.append('%.2f' %min_score)
            info.append('%.2f' %max_score)
            for _, seed, _, _, _ in predictions:
                info.append(seed)
            for _, _, _, _, tr_duration in predictions:
                info.append(tr_duration)
            for _, _, _, path, _ in predictions:
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
    if opt_partial_distribution:
        part = '-%d' %opt_partial_distribution
    else:
        part = ''
    d_name = 'distribution-%s%s%s-%d-%s-%s%s.txt' %(
        language, parser, sample, learners, test_tbid, test_type,
        part,
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

