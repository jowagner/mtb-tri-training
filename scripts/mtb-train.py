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
    oversample_ratios = [1,2,5,10,20,25,50]
    settings = []
    for constraint in range(n_seeds+len(oversample_ratios)):
        for seed_index in range(n_seeds):
            for o_index, oversample_ratio in enumerate(oversample_ratios):
                if seed_index + o_index == constraint:
                    settings.append((seed_index, oversample_ratio))
                    print(settings[-1])
    for seed_index, oversample_ratio in settings:
        seed = 101 + seed_index
        for lcode, source_experiment, source_round in [
            ('en', 'ehox--38', 5),
            ('hu', 'hhox--38', 5),
            ('ug', 'uhox--38', 5),
            ('vi', 'vh-x--36', 9),
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
                seed_files.append('%s/seed-set-%d.conllu' %(
                    workdir, i+1
                ))
                datasets.append((seed_files[-1], False, True, i+1, -1))
                synth_files.append([])
                for j in range(source_round):
                    # reminder what the different files are:
                    #   * new-candidate-set: raw knowledge transfer output
                    #   * new-selected-set: pruned down to augment size
                    #   * new-training-set: seed_set + selected data
                    synth_files[-1].append('%s/new-selected-set-%02d-%d.conllu' %(
                        workdir, j+1, i+1
                    ))
                    datasets.append((synth_files[-1][-1], False, False, i+1, j+1))
            # we forgot to keep a copy of the downsampled seed data
            # (for synthetic low-resource scenarios) but we can get
            # it from the experiments with "w" sampling
            orig_file = '%s/%sf-w--30/seed-set-1.conllu' %(workdir, source_experiment[0])
            datasets.append((orig_file, True, False, -1, -1))
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
                    exp_dir = '%s/%s/%s/%s/osr-%02d/seed-%d' %(
                        workdir, lcode, data_short_name, label_short_name,
                        oversampling_ratio, seed
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
                    f = open('%s/%s_%s-ud-train.conllu' %(exp_dir, lcode, tcode), 'wb')
                    for input_path, is_orig, is_seed_set, learner, tt_round in datasets:
                        if is_included(is_orig, is_seed_set, learner, tt_round):
                            label = labeller(is_orig, is_seed_set, learner, tt_round)
                            f.write(b'# tbemb=%s\n' %utilities.bstring(label))
                            conllu_input = open(input_path, 'rb')
                            f.write(conllu_input.read())
                            conllu_input.close()
                    f.close()
                    # create test files
                    for label in labels:
                        for input_path, test_type in [
                            (test_path, 'test'),
                            (dev_path,  'dev'),
                        ]:
                            utilities.makedirs('%s/proxy-%s' %(exp_dir, label))
                            f = open('%s/proxy-%s/%s_%s-ud-%s.conllu' %(exp_dir, label, lcode, tcode, test_type), 'wb')
                            f.write(b'# tbemb=%s\n' %utilities.bstring(label))
                            conllu_input = open(input_path, 'rb')
                            f.write(conllu_input.read())
                            conllu_input.close()
                            f.close()
                    # train parser
                    if is_multi_treebank:
                        # train multi-treebank model
                        raise NotImplementedError
                    else:
                        # train mono-treebank model
                        raise NotImplementedError

if __name__ == "__main__":
    main()

