#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

from __future__ import print_function

import hashlib
import itertools
import os
import string
import sys
import time

import conllu_dataset
import common_udpipe_future
import fasttext_udpf
import elmo_udpf
import utilities

def include_all(is_orig, is_seed_set, learner, tt_round):
    return True

def include_learners_only(is_orig, is_seed_set, learner, tt_round):
    if is_orig:
        return False
    return True

def include_plus_1(is_orig, is_seed_set, learner, tt_round):
    if is_orig:
        return True
    if is_seed_set:
        return False
    return True

def label_by_round(is_orig, is_seed_set, learner, tt_round):
    if is_orig or is_seed_set:
        return 'round_0'
    return 'round_%d' %tt_round

def label_by_learner_and_type(is_orig, is_seed_set, learner, tt_round):
    if is_orig:
        return 'manual_orig'
    if is_seed_set:
        return 'manual_%d' %learner
    return 'synthetic_%d' %learner

def label_by_learner(is_orig, is_seed_set, learner, tt_round):
    if is_orig:
        return 'manual_orig'
    return 'learner_%d' %learner

def label_by_type(is_orig, is_seed_set, learner, tt_round):
    if is_orig or is_seed_set:
        return 'manual'
    return 'synthetic'

def no_label(is_orig, is_seed_set, learner, tt_round):
    return 'na'

def main():
    n_seeds = 5
    opt_renew_all = False    # do not cover models
    opt_skip_training = True
    opt_parallel = False
    opt_sleep = False
    opt_wait_for_completion = False
    workdir = '/'.join((os.environ['PRJ_DIR'], 'workdirs'))
    tb_dir = os.environ['UD_TREEBANK_DIR']
    if 'TT_MTB_SEED_START' in os.environ:
        seed_starts = [int(os.environ['TT_MTB_SEED_START'])]
    else:
        seed_starts = [101, 201, 301]
    oversample_ratios = [1,]
    settings = []
    for constraint in range(n_seeds+len(oversample_ratios)):
        for seed_index in range(n_seeds):
            for o_index, oversample_ratio in enumerate(oversample_ratios):
                if seed_index + o_index == constraint:
                    for seed_start in seed_starts:
                        model_init_seed = '%03d' %(seed_start + seed_index)
                        settings.append((seed_index, oversample_ratio, model_init_seed))
    tasks = []
    results = {}
    for setting_idx, setting in enumerate(settings):
        seed_index, oversample_ratio, model_init_seed = setting
        print('\n\n== Setting %d of %d: Seed %s with oversampling ratio %d ==\n' %(
            setting_idx+1, len(settings), model_init_seed, oversample_ratio
        ))
        for lcode, source_experiment, source_round, tbid, tbname in [
            #'en', 'ehox--38', 5, 'en_ewt',    'English-EWT'),      # LAS 84.9
            ('en', 'eh-x--38', 5, 'en_ewt',    'English-EWT'),      # LAS 84.8
            #'hu', 'hhox--38', 5, 'hu_szeged', 'Hungarian-Szeged'), # LAS 85.8
            ('hu', 'hh-x--38', 5, 'hu_szeged', 'Hungarian-Szeged'), # LAS 85.8
            #'ug', 'uhox--38', 5, 'ug_udt',    'Uyghur-UDT'),       # LAS 71.9
            ('ug', 'uh-x--36', 9, 'ug_udt',    'Uyghur-UDT'),       # LAS 71.9
            ('vi', 'vh-x--36', 9, 'vi_vtb',    'Vietnamese-VTB'),   # LAS 68.0
        ]:
            if source_experiment[5] != '-':
                raise ValueError('Subsampling of synthethic training data not yet supported')
            n_learners = int(source_experiment[6])
            # find relevant conllu files
            # format: path, is_orig, is_seed_set, learner, tt_round
            datasets = []
            for i in range(n_learners):
                seed_file = '%s/%s/seed-set-%d.conllu' %(
                    workdir, source_experiment, i+1
                )
                datasets.append((seed_file, False, True, i+1, -1))
                for j in range(source_round):
                    # reminder what the different files are:
                    #   * new-candidate-set: raw knowledge transfer output
                    #   * new-selected-set: pruned down to augment size
                    #   * new-training-set: seed_set + selected data
                    synth_file = '%s/%s/new-selected-set-%02d-%d.conllu' %(
                        workdir, source_experiment, j+1, i+1
                    )
                    datasets.append((synth_file, False, False, i+1, j+1))
            # we forgot to keep a copy of the downsampled seed data
            # (for synthetic low-resource scenarios) but we can get
            # it from the experiments with "w" sampling
            orig_file = '%s/%sf-w--30/seed-set-1.conllu' %(workdir, source_experiment[0])
            datasets.append((orig_file, True, False, -1, -1))
            # check all files exist
            all_files_exists = True
            for filename, _, _, _, _ in datasets:
                if not os.path.exists(filename):
                    all_files_exists = False
                    print('Skipping %s due to missing %s' %(source_experiment, filename))
                    break
            if not all_files_exists:
                continue
            # create the different multi-treebank partitions
            for data_short_name, is_included in [
                #('plus1',   include_plus_1),
                ('learners', include_learners_only),
                #('all',      include_all),
            ]:
                for label_short_name, labeller in [
                    ('concat', no_label),
                    ('bytype', label_by_type),
                    ('bylearner', label_by_learner),
                    ('byboth', label_by_learner_and_type),
                    ('byround', label_by_round),
                ]:
                    exp_dir = '%s/mtb/%s/%s/%s/osr-%02d/seed-%s' %(
                        workdir, lcode, data_short_name, label_short_name,
                        oversample_ratio, model_init_seed
                    )
                    priority = int(50*setting_idx/len(settings))
                    print(exp_dir)
                    print()
                    utilities.makedirs(exp_dir)
                    tcode = 'xyz' # data_short_name + label_short_name
                    # get label set
                    labels = set()
                    for _, is_orig, is_seed_set, learner, tt_round in datasets:
                        if is_included(is_orig, is_seed_set, learner, tt_round):
                            label = labeller(is_orig, is_seed_set, learner, tt_round)
                            labels.add(label)
                    n_labels = len(labels)
                    if not n_labels:
                        print('Warning: no data selected for', short_name)
                        continue
                    is_multi_treebank = n_labels > 1
                    f = open('%s/labels.txt' %exp_dir, 'w')
                    for label in sorted(list(labels)):
                        f.write(label)
                        f.write('\n')
                    f.close()
                    # create training file
                    tr_path = '%s/%s_%s-ud-train.conllu' %(exp_dir, lcode, tcode)
                    if opt_renew_all or not os.path.exists(tr_path):
                        f = open(tr_path, 'wb')
                        for input_path, is_orig, is_seed_set, learner, tt_round in datasets:
                            if is_included(is_orig, is_seed_set, learner, tt_round):
                                label = labeller(is_orig, is_seed_set, learner, tt_round)
                                f.write(b'# tbemb=%s\n' %utilities.bstring(label))
                                conllu_input = open(input_path, 'rb')
                                conllu_data = conllu_input.read()
                                f.write(conllu_data)
                                if (is_orig or is_seed_set) and oversample_ratio > 1:
                                    for _ in range(oversample_ratio-1):
                                        f.write(conllu_data)
                                conllu_input.close()
                        f.close()
                    # create test files
                    test_details = []
                    for label in labels:
                        for input_path, test_type in [
                            ('%s/UD_%s/%s-ud-dev.conllu'  %(tb_dir, tbname, tbid), 'dev'),
                            #('%s/UD_%s/%s-ud-test.conllu' %(tb_dir, tbname, tbid), 'test'),
                        ]:
                            utilities.makedirs('%s/proxy-%s' %(exp_dir, label))
                            test_filename = '%s/proxy-%s/gold-%s_%s-ud-%s.conllu' %(
                                exp_dir, label, lcode, tcode, test_type
                            )
                            if opt_renew_all or not os.path.exists(test_filename):
                                f = open(test_filename, 'wb')
                                f.write(b'# tbemb=%s\n' %utilities.bstring(label))
                                conllu_input = open(input_path, 'rb')
                                f.write(conllu_input.read())
                                conllu_input.close()
                                f.close()
                            test_details.append((test_filename, label, lcode, tcode, test_type))

                    # submit tasks to train parsers
                    model_details = []
                    for (parser_name, parser_module) in [
                        ('elmo_udpf', elmo_udpf),
                        #('fasttext_udpf', fasttext_udpf),
                    ]:
                        model_path = '%s/%s' %(exp_dir, parser_name)
                        if not opt_skip_training and not os.path.exists(model_path):
                            tasks.append(parser_module.train(
                                tr_path, model_init_seed, model_path,
                                monitoring_datasets = test_details,
                                lcode = lcode,
                                priority = priority,
                                is_multi_treebank = is_multi_treebank,
                                submit_and_return = opt_parallel,
                            ))
                        model_details.append((model_path, parser_module, parser_name))

                    # submit tasks to make predictions for all test files and models
                    prediction_paths = []
                    for test_path, label, lcode, tcode, test_type in test_details:
                        for model_path, parser_module, parser_name in model_details:
                            prediction_path = '%s/proxy-%s/prediction-%s-%s_%s-ud-%s.conllu' %(
                                exp_dir, label, parser_name, lcode, tcode, test_type
                            )
                            if not os.path.exists(prediction_path):
                                model_ckp = model_path + '/checkpoint-inference-last.index'
                                if not opt_parallel and not os.path.exists(model_ckp):
                                    print('Cannot make predictions in sequential mode if model is not ready yet')
                                    continue
                                tasks.append(parser_module.predict(
                                    model_path, test_path, prediction_path,
                                    priority = priority,
                                    is_multi_treebank = is_multi_treebank,
                                    submit_and_return = opt_parallel,
                                    wait_for_model = True,
                                ))
                            prediction_paths.append((prediction_path, label, test_path, test_type))

                    # submit tasks to evaluate predictions
                    for prediction_path, label, test_path, test_type in prediction_paths:
                        if opt_parallel and not os.path.exists(prediction_path):
                            print('Parallel evaluation not supported. Please re-run this script when all predictions are ready.')
                            continue
                        if not os.path.exists(prediction_path):
                            print('Cannot evaluate predictions in sequential mode if prediction is not ready yet.')
                            continue
                        score = conllu_dataset.evaluate(
                            prediction_path, test_path
                        )
                        key = (
                            test_type,
                            source_experiment, source_round,
                            oversample_ratio,
                            data_short_name, label_short_name,
                            label,
                        )
                        if not key in results:
                            results[key] = []
                        # add seed to score info
                        score = tuple(list(score) + [model_init_seed])
                        results[key].append(score)

                    print()
                    print('# tasks so far:', len(tasks))
                    if opt_sleep and setting_idx == 0:
                        print('sleeping 1 minute')
                        time.sleep(60)
                    elif opt_sleep:
                        print('sleeping 20 minutes')
                        time.sleep(1200)
                    print()

                    if results:
                        print()
                        part1_keys = set()
                        part2_keys = set()
                        part3_keys = set()
                        for key in sorted(list(results.keys())):
                            scores = results[key]
                            scores.sort()
                            short_scores = ', '.join(['%.1f' %(score[0]) for score in scores])
                            used_seeds   = ', '.join([score[-1]          for score in scores])
                            print(key, short_scores, 'seeds:', used_seeds)
                            part1_keys.add(key[0])
                            part2_keys.add(key[1:3])
                            part3_keys.add(key[3:])
                        print()
                        for key1 in sorted(list(part1_keys)):
                            for key3 in sorted(list(part3_keys)):
                                avg_scores = []
                                pick_from = []
                                for key2 in part2_keys:
                                    key = (key1,) + key2 + key3
                                    if not key in results:
                                        if key3[-1] in ('round_6', 'round_7', 'round_8', 'round_9'):
                                            key = (key1,) + key2 + key3[:-1] + ('round_5',)
                                            if not key in results:
                                                break
                                        else:
                                            break
                                    scores = [score[0] for score in results[key]]
                                    pick_from.append(scores)
                                if len(pick_from) < len(part2_keys):
                                    continue
                                for combination in itertools.product(*pick_from):
                                    avg_scores.append(sum(combination)/len(combination))
                                score_stats = utilities.get_score_stats(avg_scores)
                                # min_score, score025, score250, median, score750, score975, num_scores
                                short_scores = ', '.join(['%.1f' %score for score in score_stats])
                                print(key1, key3, short_scores, '(over %d averages)' %len(avg_scores))
                        print()

    # we must wait for training and prediction tasks to finish in order for
    # temporary files to be deleted
    if opt_parallel and opt_wait_for_completion:
        common_udpipe_future.wait_for_tasks(tasks)

if __name__ == "__main__":
    main()

