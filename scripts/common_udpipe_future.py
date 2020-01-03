#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

from __future__ import print_function

import os
import subprocess
import sys

def get_training_schedule(epochs = 60):
    ''' return a udpipe-future learning rate schedule
        and epochs specification like
        "30:1e-3,5:6e-4,5:4e-4,5:3e-4,5:2e-4,10:1e-4"
        adjusted to the given number of epochs
    '''
    if epochs < 1:
        raise ValueError('Need at least 1 epoch to train a model.')
    ref_remaining = 60
    epochs_remaining = epochs
    components = []
    for ref_count, learning_rate in [
        (30, '1e-3'),
        ( 5, '6e-4'),
        ( 5, '4e-4'),
        ( 5, '3e-4'),
        ( 5, '2e-4'),
        (10, '1e-4'),
    ]:
        n = int(epochs_remaining*ref_count/ref_remaining)
        ref_remaining -= ref_count
        epochs_remaining -= n
        if n > 0:
            components.append('%d:%s' %(n, learning_rate))
    return ','.join(components)

def memory_error(model_dir):
    if not os.path.exists('%s/stderr.txt' %model_dir):
        # problem is somewhere in the wrapper script
        return False
    f = open('%s/stderr.txt' %model_dir)
    found_oom = False
    while True:
        line = f.readline()
        if not line:
            break
        if 'ran out of memory trying to allocate' in line:
            found_oom = True
            break
    f.close()
    return found_oom

def incomplete(model_dir):
    if not os.path.exists('%s/checkpoint' %model_dir):
        return True
    return False

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()
