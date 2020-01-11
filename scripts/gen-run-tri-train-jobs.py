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

if len(sys.argv) > 1 and sys.argv[1] == 'ichec':
    template = open('template-ichec-2x.job', 'rb').read()
    augment_offset = 1
    augment_step = 4
    gpu_list = [
        ('v100', 'v100'),
    ]
else:
    template = open('run-tri-train.job', 'rb').read()
    augment_offset = 0
    augment_step = 2
    gpu_list = [
        ('tesla', 'tesla'),
        ('rtx',   'rtx2080ti'),
    ]

if not os.path.exists('jobs'):
    os.mkdir('jobs')

setting2modelseedsuffix = {}
modelseedsuffix2setting = []

def get_modelseedsuffix(setting):
    global setting2modelseedsuffix
    global modelseedsuffix2setting
    try:
        retval = setting2modelseedsuffix[setting]
    except KeyError:
        retval = len(modelseedsuffix2setting)
        setting2modelseedsuffix[setting] = retval
        modelseedsuffix2setting.append(setting)
    return retval

aug2iterations = [24, 20, 17, 16, 14, 12, 9, 7, 5, 3]

for augment_size_code in range(augment_offset,10,augment_step):
    augsize = int(0.5+5*(2.0**0.5)**augment_size_code)
    augsize2 = int(0.5+5*(2.0**0.5)**(augment_size_code+2))
    subsetsize = 16 * augsize  # max(min(320, 11 * augsize), 7*augsize)
    for major_code, more_options in [
        (3, ''),
        #(5, '--learners 5'),
        #(9, '--learners 9'),
    ]:
        seed = '%d%d' %(major_code, augment_size_code)
        seed2 = '%d%d' %(major_code, augment_size_code+2)
        for ovs_code, ovs_options in [
            ('-', ''),
            ('o', '--oversample'),
        ]:
            for wrpl_code, wrpl_options in [
                ('-', ''),
                ('w', '--without-replacement'),
                ('x', '--without-replacement --seed-size "250%"'),
            ]:
                for disa_code, disa_options in [
                    ('-', ''),
                    #('d', '--min-learner-disagreement 1'),
                    #('e', '--min-learner-disagreement 2'),
                ]:
                    for decay_code, decay_options in [
                        ('-', ''),
                        ('v', '--last-k 1'),
                        #('y', '--last-k 5'),
                        ('z', '--last-decay 0.5'),
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
                            #'f', 'fr', 'fr_gsd',    160444000, ''),
                            #'g', 'ga', 'ga_9010idt', 20462403, ''),
                            ('h', 'hu', 'hu_szeged', 168359253, ''),
                            ('u', 'ug', 'ug_udt',      2537468, ''),
                            ('v', 'vi', 'vi_vtb',    189658820, ''),
                        ]:
                            #iterations = min(24, int(0.5+0.002*unlabelled_size/augsize))
                            iterations = aug2iterations[augment_size_code]
                            if wrpl_code != 'x':
                                iterations = 0
                            for parser_code, model_module in [
                                #('a', 'allennlp'),
                                ('f', 'udpipe_future'),
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
                                if parser_code in 'afum':
                                    model_keyword_options = ''
                                else:
                                    model_keyword_options = '--model-keyword lcode %s' %lcode
                                name = '%s%s%s%s%s%s%s' %(short_lcode, parser_code, ovs_code, wrpl_code, disa_code, decay_code, seed)
                                name2 = '%s%s%s%s%s%s%s' %(short_lcode, parser_code, ovs_code, wrpl_code, disa_code, decay_code, seed2)
                                for gpu_short, gpu_name in gpu_list:
                                    f = open('jobs/%s-%s.job' %(name, gpu_short), 'wb')
                                    f.write(template %locals())
                                    f.close()

