#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

import os
import sys

repeat_run = 0
while len(sys.argv) > 1 and sys.argv[1] == '--repeat':
    # repeat run with different main seed
    repeat_run += 1
    del sys.argv[1]

require_workdir_to_preexist = False   # True = only create jobs for jobs that have been run before

job_dir = 'mbert-tuning-jobs'

template = open('template-mbert-tuning.job', 'rt').read()

if not os.path.exists(job_dir):
    os.mkdir(job_dir)

seed = '%d' %(100+repeat_run)

for pooling_code, pooling in [
    ('avg', 'average'),
    ('fst', 'first'),
    ('lst', 'last'),
    ('max', 'max'),
    ('z50', 'binomial50'),
    ('z30', 'binomial30'),
    ('z70', 'binomial70'),
]:
    for layer_code, layer, expand_to_values in [
        ('L08',  8, ('0768', 0)),
        ('L09',  9, ('0768', 0)),
        ('L10', 10, ('0768', 0)),
        ('L11', 11, ('0768', 0)),
        ('L12', 12, ('0768', 0)),
        ('LA4', -4, ('0768', 0)),
        ('LA4', -4, ('1024', 1024)),
        ('LA5', -5, ('0768', 0)),
        ('LA5', -5, ('1024', 1024)),
    ]:
        expand_to_code, expand_to = expand_to_values
        for short_lcode, lcode, tbid, lang_options in [
            #'c', 'cs', 'cs_pdt',     '--simulate-size 20k --simulate-seed 42'),
            #'d', 'de', 'de_gsd',     '--simulate-size 20k --simulate-seed 42'),
            ('e', 'en', 'en_ewt',     '--simulate-size 20k --simulate-seed 42'),
            #'f', 'fr', 'fr_gsd',     ''),
            #'g', 'ga', 'ga_9010idt', ''),
            ('h', 'hu', 'hu_szeged',  ''),
            ('u', 'ug', 'ug_udt',     ''),
            ('v', 'vi', 'vi_vtb',     ''),
        ]:
            name = '%(pooling_code)s-%(layer_code)s-%(expand_to_code)s-%(lcode)s' %locals()
            if require_workdir_to_preexist:
                workdir = '/'.join([os.environ['PRJ_DIR'], 'workdirs', name])
                if not os.path.exists(workdir):
                    continue
            f = open('%s/%s.job' %(job_dir, name), 'wt')
            f.write(template %locals())
            f.close()

