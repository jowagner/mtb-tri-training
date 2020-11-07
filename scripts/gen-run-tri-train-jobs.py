#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

# TODO: make all parts work with Python3

from __future__ import print_function

import os
import sys

if len(sys.argv) > 1 and sys.argv[1].startswith('--high-prio'):
    opt_high_priority = True
    del sys.argv[1]
else:
    opt_high_priority = False

if len(sys.argv) > 1 and sys.argv[1] == '--repeat':
    # repeat run with different main seed
    repeat_run = 1
    del sys.argv[1]
else:
    repeat_run = 0

mini_option = ''
max_iterations = 9999
if len(sys.argv) > 1 and sys.argv[1].startswith('--baseline'):
    job_dir = 'baseline-jobs'
    max_iterations = 0
    del sys.argv[1]
elif len(sys.argv) > 1 and sys.argv[1].startswith('--mini'):
    job_dir = 'mini-jobs'
    mini_option = '--max-model-training 2'
    del sys.argv[1]
elif len(sys.argv) > 1 and sys.argv[1].startswith('--eval'):
    job_dir = 'eval-jobs'
    mini_option = '--max-model-training 0'
    del sys.argv[1]
else:
    job_dir = 'jobs'

if len(sys.argv) > 1 and sys.argv[1].startswith('ichec'):
    template = open('template-ichec-2x.job', 'rb').read()
    if sys.argv[1][-1] == 'c':
        augment_size_codes = [1,5,]
    else:
        augment_size_codes = [3,7,]
    gpu_list = [
        ('v100', 'v100'),
    ]
else:
    template = open('run-tri-train.job', 'rb').read()
    augment_size_codes = [0,2,4,6,8,10,12,14]
    gpu_list = [
        #('tesla', 'tesla'),
        #('rtx',   'rtx2080ti'),
        ('task',   'none'),
    ]
    if len(sys.argv) > 1 and len(sys.argv[1]) == 1:
        augment_size_codes = []
        augment_size_codes.append(int(sys.argv[1], 16))
    if len(sys.argv) > 1 and sys.argv[1] == 'big':
        augment_size_codes = augment_size_codes[4:]
    if len(sys.argv) > 1 and sys.argv[1] == 'grove':
        augment_size_codes = [0,2,4]
    if len(sys.argv) > 1 and sys.argv[1] == 'grove+':
        augment_size_codes = [3,5]
    if len(sys.argv) > 1 and sys.argv[1] == 'boole':
        augment_size_codes = [6,8]
    if len(sys.argv) > 1 and sys.argv[1] == 'boole+':
        augment_size_codes = [7,9]
    if len(sys.argv) > 1 and sys.argv[1] == 'okia':
        augment_size_codes = [0,]
    if len(sys.argv) > 1 and sys.argv[1] == 'okia+':
        augment_size_codes = [1,]

if not os.path.exists(job_dir):
    os.mkdir(job_dir)

setting2modelseedsuffix = {}
modelseedsuffix2setting = []

def get_modelseedsuffix(setting):
    global setting2modelseedsuffix
    global modelseedsuffix2setting
    try:
        retval = setting2modelseedsuffix[setting]
    except KeyError:
        retval = len(modelseedsuffix2setting)
        retval += 200  # skip values used in previous version
        setting2modelseedsuffix[setting] = retval
        modelseedsuffix2setting.append(setting)
    return retval

#                      0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
aug2iterations      = [24, 22, 20, 18, 16, 14, 12, 10,  8,  6,  4,  3,  2,  1,  1,  1]

aug2last_iterations = [24, 21, 17, 15, 14, 11,  9,  7,  5,  4,  3,  2,  2,  2,  2,  2]    # for --round-priority

if opt_high_priority:
    aug2last_iterations = map(lambda x: 10*x, aug2last_iterations)


for augment_size_code in augment_size_codes:
    augsize = int(0.5+5*(2.0**0.5)**augment_size_code)
    augsize2 = int(0.5+5*(2.0**0.5)**(augment_size_code+2))
    subsetsize = 16 * augsize  # max(min(320, 11 * augsize), 7*augsize)
    for major_code, more_options in [
        (3, ''),
        #(5, '--learners 5'),
        #(9, '--learners 9'),
    ]:
        seed = '%d%d' %(major_code+repeat_run, augment_size_code)
        seed2 = '%d%d' %(major_code+repeat_run, augment_size_code+2)
        for ovs_code, ovs_options in [
            ('-', ''),
            ('o', '--oversample'),
        ]:
            for wrpl_code, wrpl_options in [
                ('-', ''),
                ('w', '--without-replacement'),
                ('x', '--without-replacement --seed-size "250%"'),
                ('t', '--without-replacement --seed-size "300%"'),
            ]:
                for disa_code, disa_options in [
                    #('a', '--all-knowledge-transfers'),  # not implemented yet
                    ('-', ''),
                    ('d', '--min-learner-disagreement 1'),
                    #('e', '--min-learner-disagreement 2'),
                ]:
                    for decay_code, decay_options in [
                        ('-', ''),
                        ('v', '--last-k 1'),
                        ('y', '--last-k 5'),
                        ('z', '--last-decay 0.5'),
                        ('o', '--last-decay 0.71'),
                        ('a', '--all-labelled-data'),
                        ('u', '--all-labelled-data --last-k 1 --check-error-rate'),
                        ('f', '--all-labelled-data --last-k 5'),
                        ('r', '--all-labelled-data --last-decay 0.5'),
                        ('s', '--all-labelled-data --last-decay 0.71'),
                    ]:
                        #if decay_code != '-' and augment_size_code < 6:
                        #    continue
                        modelseedsuffix = get_modelseedsuffix(
                            ovs_code+wrpl_code+disa_code+decay_code
                        )
                        for short_lcode, lcode, tbid, unlabelled_size, lang_options in [
                            #'c', 'cs', 'cs_pdt',    160444000, '--simulate-size 20k --simulate-seed 42'),
                            #'d', 'de', 'de_gsd',    160444000, '--simulate-size 20k --simulate-seed 42'),
                            ('e', 'en', 'en_ewt',    153878772, '--simulate-size 20k --simulate-seed 42'),
                            #('t', 'en', 'en_lines',   204607, '--no-test-unlabelled'),  # manually change template to use en_ewt as unlabelled data
                            #'f', 'fr', 'fr_gsd',    160444000, ''),
                            #'g', 'ga', 'ga_9010idt', 20462403, ''),
                            ('h', 'hu', 'hu_szeged', 168359253, ''),
                            ('u', 'ug', 'ug_udt',      2537468, ''),
                            ('v', 'vi', 'vi_vtb',    189658820, ''),
                        ]:
                            #iterations = min(24, int(0.5+0.002*unlabelled_size/augsize))
                            iterations = aug2iterations[augment_size_code]
                            last_iterations = aug2last_iterations[augment_size_code]
                            iterations = min(max_iterations, iterations)
                            for parser_code, model_module in [
                                #('a', 'allennlp'),
                                ('f', 'udpipe_future'),
                                #('t', 'udpipe_future'),
                                ('g', 'fasttext_udpf'),
                                ('h', 'elmo_udpf'),
                                #('i', 'mbert_udpf'),
                                #('u', 'uuparser'),
                                #('v', 'fasttext_uup'),
                                #('w', 'elmo_uup'),
                                #('m', 'mixed'),
                                #('n', 'fasttext_mx'),
                                #('o', 'elmo_mx'),
                            ]:
                                if parser_code in 'aftum':
                                    model_keyword_options = ''
                                else:
                                    model_keyword_options = '--model-keyword lcode %s' %lcode
                                name = '%s%s%s%s%s%s%d%X' %(
                                    short_lcode, parser_code, ovs_code, wrpl_code,
                                    disa_code, decay_code, major_code+repeat_run,
                                    augment_size_code
                                )
                                name2 = '%s%s%s%s%s%s%d%X' %(
                                    short_lcode, parser_code, ovs_code, wrpl_code,
                                    disa_code, decay_code, major_code+repeat_run,
                                    augment_size_code+2
                                )
                                for gpu_short, gpu_name in gpu_list:
                                    f = open('%s/%s-%s.job' %(job_dir, name, gpu_short), 'wb')
                                    f.write(template %locals())
                                    f.close()

