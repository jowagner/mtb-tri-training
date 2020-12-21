#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

from __future__ import print_function

import importlib
import hashlib
import os
import sys

import utilities

def print_usage():
    print('Usage: %s [options] < main-comparison.tsv > baseline-test-results.txt' %(os.path.split(sys.argv[0])[-1]))
    print("""
Options:

    --average   NUM         Average over NUM runs of the combiner
                            (default: 21)

    --dataset-module  NAME  handle data sets using the module NAME;
                            combine with --help to see module-specific options
                            (default: conllu_dataset)

    --dataset-basedir  PATH  Use PATH with the dataset module's load function
                            (Default: locate datasets in some other way)

    --learners  NUM         Use NUM learners from input file
                            (default: 3)

    --load-data-keyword  KEY  VALUE
                            Pass additional key-value pair as keyword
                            arguments to the load() function of each dataset
                            module when loading the development or test
                            data.
                            Can be specified multiple times to specify more
                            than one non-standard argument.

    --models  NAMES         Colon- or space-separated list of models to evaluate
                            (default: udpf:elmo)

    --test-types  NAMES     Colon- or space-separated list of test set types
                            to evaluate
                            (default: dev:test)

    --workdirs  DIR         Path to parent folder of experiment directories
                            (default: workdirs)

""")

def main():
    opt_help  = False
    opt_debug  = False
    opt_verbose = False
    opt_average = 21
    opt_dataset_basedir = None
    opt_dataset_module = 'conllu_dataset'
    opt_learners = 3
    opt_load_data_kwargs = {}
    opt_models     = ('udpf', 'elmo')
    opt_test_types = ('dev', 'test')
    opt_workdirs = 'workdirs'

    while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
        option = sys.argv[1]
        option = option.replace('_', '-')
        del sys.argv[1]
        if option in ('--help', '-h'):
            opt_help = True
            break
        elif option == '--average':
            opt_average = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--dataset-basedir':
            opt_dataset_basedir = sys.argv[1]
            del sys.argv[1]
        elif option == '--dataset-module':
            opt_dataset_module = sys.argv[1]
            del sys.argv[1]
        elif option == '--learners':
            opt_learners = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--load-data-keyword':
            key = sys.argv[1]
            value = sys.argv[2]
            opt_load_data_kwargs[key] = value
            del sys.argv[1]   # consume two args
            del sys.argv[1]
        elif option == '--models':
            opt_models = sys.argv[1].replace(':', ' ').split()
            del sys.argv[1]
        elif option == '--test-types':
            opt_test_types = sys.argv[1].replace(':', ' ').split()
            del sys.argv[1]
        elif option == '--workdirs':
            opt_workdirs = sys.argv[1]
            del sys.argv[1]
        elif option == '--verbose':
            opt_verbose = True
        elif option == '--debug':
            opt_debug = True
        else:
            print('Unsupported or not yet implemented option %s' %option)
            opt_help = True
            break

    if len(sys.argv) != 1:
        opt_help = True

    if opt_help:
        print_usage()
        sys.exit(0)

    dataset_module = importlib.import_module(opt_dataset_module)
    filename_extension = dataset_module.get_filename_extension()

    header = sys.stdin.readline().rstrip().split('\t')
    lang_column = header.index('Language')
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        row = line.rstrip().split('\t')
        language = row[lang_column]
        if language == 'Average':
            break
        for test_type in opt_test_types:
            for parser in opt_models:
                print('\n== %s %s %s ==\n' %(language, test_type, parser))
                # (1) find predictions and treebank IDs
                exp_codes = []
                tbids = set()
                tbid_and_learner2best_candidate = {}
                for learner_index in range(opt_learners):
                    learner_rank = learner_index + 1
                    exp_code_column = header.index('b-%s-L%d' %(parser, learner_rank))
                    exp_code = row[exp_code_column]
                    exp_codes.append(exp_code)
                    exp_dir = '/'.join((opt_workdirs, exp_code))
                    # find the prediction file to use
                    for entry in os.listdir(exp_dir):
                        # 0          1  2 3      4   5
                        # prediction-00-1-en_ewt-dev-p8UpLh6rKISx0mXAfQ1j.conllu.bz2
                        if not entry.startswith('prediction-00-%d-' %learner_rank):
                            continue
                        if not entry.endswith('.conllu.bz2'):
                            continue
                        fields = entry.split('-')
                        # exclude files with -incomplete or -oom* suffix
                        if len(fields) != 6:
                            continue
                        # check test type
                        if fields[4] != test_type:
                            continue
                        # found a candidate
                        candidate_path = '%s/%s' %(exp_dir, entry)
                        tbid = fields[3]
                        # prefer the newest model
                        priority = os.path.getmtime(candidate_path)
                        candidate = (priority, candidate_path)
                        key = (tbid, learner_index)
                        #print('Found candidate', candidate, 'with priority', priority, 'and key', key)
                        if not key in tbid_and_learner2best_candidate \
                        or candidate > tbid_and_learner2best_candidate[key]:
                            tbid_and_learner2best_candidate[key] = candidate
                            #print('Updated best candidate')
                        # record treebank ID
                        tbids.add(tbid)
                usable_tbids = set()
                for tbid in tbids:
                    usable = True
                    for learner_index in range(3):
                        key = (tbid, learner_index)
                        if not key in tbid_and_learner2best_candidate:
                            usable = False
                            break
                    if usable:
                        usable_tbids.add(tbid)
                if not usable_tbids:
                    print('Warning: missing predictions', tbid_and_learner2best_candidate)
                for tbid in usable_tbids:
                    print('Evaluating on', tbid, test_type)
                    _, dev_data, test_data = dataset_module.load(
                        tbid, load_test = test_type == 'test',
                        dataset_basedir = opt_dataset_basedir,
                        **opt_load_data_kwargs
                    )
                    if test_type == 'test':
                        gold_path = test_data.filename
                    else:
                        gold_path = dev_data.filename
                    pred_paths = []
                    pred_fingerprints = []
                    print('Combiner input:')
                    for learner_index in range(opt_learners):
                        key = (tbid, learner_index)
                        _, pred_path = tbid_and_learner2best_candidate[key]
                        pred_paths.append(pred_path)
                        pred_fingerprints.append(hashlib.sha512(pred_path).hexdigest())
                        print('%6d\t%s' %(learner_index+1, pred_path))
                    ensemble_fingerprint = utilities.hex2base62(hashlib.sha512(
                        ':'.join(pred_fingerprints)
                    ).hexdigest())
                    output_paths = []
                    score_and_run = []
                    print('Running combiner %d times...' %opt_average)
                    sys.stdout.flush()
                    for run_index in range(opt_average):
                        output_path = '%s/baseline-tt-sim-%s-%s-E-%d-%s-%s-%s%s' %(
                            opt_workdirs,
                            tbid.split('_')[0], parser,
                            run_index,
                            tbid, test_type,
                            ensemble_fingerprint[:20],
                            filename_extension
                        )
                        dataset_module.combine(pred_paths, output_path,
                            seed = '%d' %(100+run_index),
                            verbose = False,
                        )
                        score, score_s, eval_path = dataset_module.evaluate(
                            output_path, gold_path,
                            verbose = False,
                        )
                        output_paths.append((output_path, eval_path))
                        score_and_run.append((score, run_index))
                    average_score = get_average_score_and_cleanup(
                        score_and_run,
                        output_paths,
                        keep_median_files = True,
                        print_range = True,
                        print_stddev = True,
                        print_median = True,
                        print_average = True,
                    )
                    print('Score: %.9f' %average_score)

def get_average_score_and_cleanup(
    score_and_run, output_paths,
    keep_median_files = False,
    print_range = False,
    print_stddev = False,
    print_median = False,
    print_average = False,
):
    scores = map(lambda x: x[0], score_and_run)
    average_score = sum(scores) / float(len(scores))
    if print_range:
        min_score = min(scores)
        print('Lowest score: %.9f (%.2f)' %(min_score, min_score))
    if print_average:
        print('Average score: %.9f (%.2f)' %(average_score, average_score))
    if print_range:
        max_score = max(scores)
        print('Highest score: %.9f (%.2f)' %(max_score, max_score))
        print('Score range: %.9f' %(max_score-min_score))
    if print_stddev:
        sq_errors = []
        for score in scores:
            error = average_score - score
            sq_errors.append(error**2)
        n = len(scores)
        std_dev = (sum(sq_errors)/float(n))**0.5
        print('Population std dev = %.9f (%.2f)' %(std_dev, std_dev))
        std_dev = (sum(sq_errors)/(n-1.0))**0.5
        print('Simple sample std dev = %.9f (%.2f)' %(std_dev, std_dev))
        std_dev = (sum(sq_errors)/(n-1.5))**0.5
        print('Approximate std dev = %.9f (%.2f)' %(std_dev, std_dev))
        std_dev = (sum(sq_errors)/(n-1.5+1.0/(8.0*(n-1.0))))**0.5
        print('More accurate std dev = %.9f (%.2f)' %(std_dev, std_dev))
    if keep_median_files or print_median:
        # find median
        score_and_run.sort()
        while len(score_and_run) > 2:
            # remove lowest and highest scoring result
            # to move towards the middle
            for obsolete_index in (-1, 0):
                _, run_index = score_and_run[obsolete_index]
                del score_and_run[obsolete_index]
                # clean up output files
                output_path, eval_path = output_paths[run_index]
                os.unlink(output_path)
                if eval_path:
                    os.unlink(eval_path)
        if len(score_and_run) == 2:
            median_score = (score_and_run[0][0]+score_and_run[1][0])/2.0
            _, run_index = score_and_run[0]
            del score_and_run[0]
            # clean up output files of first of the two median elements
            output_path, eval_path = output_paths[run_index]
            os.unlink(output_path)
            if eval_path:
                os.unlink(eval_path)
        else:
            median_score = score_and_run[0][0]
        if print_median:
            print('Median score: %.9f (%.2f)' %(median_score, median_score))
    if not keep_median_files:
        for _, run_index in score_and_run:
            output_path, eval_path = output_paths[run_index]
            os.unlink(output_path)
            if eval_path:
                os.unlink(eval_path)
    return average_score

if __name__ == "__main__":
    main()

