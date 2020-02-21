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
import os
import string
import sys
import time

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
    n_seeds = 11
    workdir = 'workdirs'
    tb_dir = os.environ['UD_TREEBANK_DIR']
    oversample_ratios = [1,2,5,10,20,25,50]
    settings = []
    for constraint in range(n_seeds+len(oversample_ratios)):
        for seed_index in range(n_seeds):
            for o_index, oversample_ratio in enumerate(oversample_ratios):
                if seed_index + o_index == constraint:
                    settings.append((seed_index, oversample_ratio))
    tasks = []
    for setting_idx, setting in enumerate(settings):
        seed_index, oversample_ratio = setting
        model_init_seed = 101 + seed_index
        print('== Setting %d of %d: Seed %d with oversampling ratio %d ==' %(
            setting_idx+1, len(settings), seed, oversample_ratio
        ))
        for lcode, source_experiment, source_round, tbid, tbname in [
            ('en', 'ehox--38', 5, 'en_ewt',    'English-EWT'),
            ('hu', 'hhox--38', 5, 'hu_szeged', 'Hungarian-Szeged'),
            ('ug', 'uhox--38', 5, 'ug_udt',    'Uyghur-UDT'),
            ('vi', 'vh-x--36', 9, 'vi_vtb',    'Vietnamese-VTB'),
        ]:
            if source_experiment[5] != '-':
                raise ValueError('Subsampling of synthethic training data not yet supported')
            n_learners = int(source_experiment[6])
            # find relevant conllu files
            seed_files = []
            synth_files = []
            # format: path, is_orig, is_seed_set, learner, tt_round
            datasets = []
            for i in range(n_learners):
                seed_files.append('%s/%s/seed-set-%d.conllu' %(
                    workdir, source_experiment, i+1
                ))
                datasets.append((seed_files[-1], False, True, i+1, -1))
                synth_files.append([])
                for j in range(source_round):
                    # reminder what the different files are:
                    #   * new-candidate-set: raw knowledge transfer output
                    #   * new-selected-set: pruned down to augment size
                    #   * new-training-set: seed_set + selected data
                    synth_files[-1].append('%s/%s/new-selected-set-%02d-%d.conllu' %(
                        workdir, source_experiment, j+1, i+1
                    ))

                    datasets.append((synth_files[-1][-1], False, False, i+1, j+1))
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
                ('plus1',   include_plus_1),
                ('learners', include_learners_only),
                ('all',      include_all),
            ]:
                for label_short_name, labeller in [
                    ('concat', no_label),
                    ('bytype', label_by_type),
                    ('bylearner', label_by_learner),
                    ('byboth', label_by_learner_and_type),
                    ('byround', label_by_round),
                ]:
                    exp_dir = '%s/mtb/%s/%s/%s/osr-%02d/seed-%d' %(
                        workdir, lcode, data_short_name, label_short_name,
                        oversample_ratio, seed
                    )
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
                    for label in labels:
                        f.write(label)
                        f.write('\n')
                    f.close()
                    # create training file
                    tr_path = '%s/%s_%s-ud-train.conllu' %(exp_dir, lcode, tcode)
                    f = open(tr_path, 'wb')
                    for input_path, is_orig, is_seed_set, learner, tt_round in datasets:
                        if is_included(is_orig, is_seed_set, learner, tt_round):
                            label = labeller(is_orig, is_seed_set, learner, tt_round)
                            f.write(b'# tbemb=%s\n' %utilities.bstring(label))
                            conllu_input = open(input_path, 'rb')
                            f.write(conllu_input.read())
                            conllu_input.close()
                    f.close()
                    # create test files
                    test_details = []
                    for label in labels:
                        for input_path, test_type in [
                            ('%s/UD_%s/%s-ud-dev.conllu'  %(tb_dir, tbname, tbid), 'dev'),
                            ('%s/UD_%s/%s-ud-test.conllu' %(tb_dir, tbname, tbid), 'test'),
                        ]:
                            utilities.makedirs('%s/proxy-%s' %(exp_dir, label))
                            test_filename = '%s/proxy-%s/gold-%s_%s-ud-%s.conllu' %(
                                exp_dir, label, lcode, tcode, test_type
                            )
                            f = open(test_filename, 'wb')
                            f.write(b'# tbemb=%s\n' %utilities.bstring(label))
                            conllu_input = open(input_path, 'rb')
                            f.write(conllu_input.read())
                            conllu_input.close()
                            f.close()
                            test_details.append((test_filename, label, lcode, tcode, test_type))

                    # submit tasks to train parsers
                    model_details = []
                    if is_multi_treebank:
                        # train multi-treebank model
                        pass # raise NotImplementedError
                        # TODO: can we use is_multi_treebank as below?
                    else:
                        # train mono-treebank model, if it doesn't exist yet
                        for (parser_name, parser_module) in [
                            ('elmo_udpf', elmo_udpf),
                            ('fasttext_udpf', fasttext_udpf),
                        ]:
                            model_path = '%s/%s' %(exp_dir, parser_name)
                            if not os.path.exists(model_path):
                                tasks.append(parser_module.train(
                                    tr_path, model_init_seed, model_path,
                                    epoch_selection_set = None,
                                    monitoring_datasets = test_details,
                                    priority = 0,
                                    is_multi_treebank = is_multi_treebank,
                                    submit_and_return = True,
                                ))
                            model_details.append((model_path, parser_module, parser_name))

                    # submit tasks to make predictions for all test files and models
                    prediction_paths = []
                    for test_path, label, lcode, tcode, test_type in test_details:
                        for model_path, parser_module, parser_name in model_details:
                            # the model must be ready
                            # before the prediction task can start
                            requires = [model_path]
                            prediction_path = '%s/proxy-%s/prediction-%s-%s_%s-ud-%s.conllu' %(
                                exp_dir, label, parser_name, lcode, tcode, test_type
                            )
                            if is_multi_treebank:
                                pass
                            else:
                                pass
                            prediction_paths.append((prediction_path, test_path))

                    # submit tasks to evaluate predictions
                    for prediction_path, test_path in prediction_paths:
                        requires = [prediction_path]

    # we must wait for training and prediction tasks to finish in order for
    # temporary files to be deleted
    common_udpipe_future.wait_for_tasks(tasks)

if __name__ == "__main__":
    main()

