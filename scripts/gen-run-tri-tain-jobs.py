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

template = open('run-tri-train.job', 'rb').read()

for augment_size_code in range(10):
    augsize = int(0.5+5*(2.0**0.5)**augment_size_code)
    subsetsize = 16 * augsize
    iterations = int(0.5+2*204.585/augsize)
    for major_code, more_options in [
        (3, ''),
        (5, '--learners 5'),
        #(9, '--learners 9'),
    ]:
        seed = '%d%d' %(major_code, augment_size_code)
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
                    ('d', '--min-learner-disagreement 1'),
                    #('e', '--min-learner-disagreement 2'),
                ]:
                    for decay_code, decay_options in [
                        ('-', ''),
                        ('y', '--last-k 5'),
                        #('z', '--last-decay 0.5'),
                    ]:
                        for short_lcode, lcode, tbid  in [
                            #('c', 'cs', 'cs_pdt'),
                            #('d', 'de', 'de_gsd'),
                            #('e', 'en', 'en_ewt'),
                            #('f', 'fr', 'fr_gsd'),
                            ('g', 'ga', 'ga_idt'),
                        ]:
                            for parser_code, model_module in [
                                #('a', 'allennlp'),
                                ('f', 'udpipe_future'),
                                ('g', 'fasttext_udpf'),
                                ('h', 'elmo_udpf'),
                                #('i', 'mbert_udpf'),
                                #('u', 'uuparser'),
                                #('v', 'fasttext_uup'),
                                #('x', 'elmo_uup'),
                                #('m', 'mixed'),
                                #('n', 'fasttext_mx'),
                                #('o', 'elmo_mx'),
                            ]:
                                name = '%s%s%s%s%s%s%s' %(short_lcode, parser_code, ovs_code, wrpl_code, disa_code, decay_code, seed)
                                f = open('r-%s.job' %name, 'wb')
                                f.write(template %locals())
                                f.close()

