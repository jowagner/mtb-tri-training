#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# runs with both Python 2 and Python 3

import os
import sys

workdirs = './all'

while True:
    line = sys.stdin.readline()
    if not line:
        break
    fields = line.rstrip().split('\t')
    new_fields = fields[:-3]
    for learner_rank, prediction in enumerate(fields[-3:], start=1):
        last_sep = prediction.rfind('/')
        folder = workdirs + '/' + prediction[:last_sep]
        model = None
        prefix = 'model-00-%d' %learner_rank
        for entry in os.listdir(folder):
            if entry.startswith(entry):
                try:
                    start_t = os.path.getmtime('%s/%s/training.start' %(folder, entry))
                except:
                    start_t = None
                try:
                    end_t = os.path.getmtime('%s/%s/training.end' %(folder, entry))
                except:
                    end_t = None
                if start_t and end_t:
                    model = entry
                    duration = end_t - start_t
        if not model:
            duration = 'unknown'
        else:
            duration = '%.3f' %duration
        new_fields.append(duration)
    for prediction in fields[-3:]:
        new_fields.append(prediction)
    sys.stdout.write('\t'.join(new_fields))
    sys.stdout.write('\n')
