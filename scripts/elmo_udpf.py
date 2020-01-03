#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
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

import common_udpipe_future

def train(
    dataset_filename, seed, model_dir,
    epoch_selection_dataset = None,
    monitoring_datasets = [],
    lcode = None,
    batch_size = 32,
):
    if lcode is None:
        raise ValueError('Missing lcode; use --module-keyword to specify a key-value pair')
    if epoch_selection_dataset:
        raise ValueError('Epoch selection not supported with udpipe-future.')
    command = []
    command.append('./elmo_udpf-train.sh')
    command.append(dataset_filename)
    command.append(lcode)
    if seed is None:
        raise NotImplementedError
    command.append(seed)
    command.append(model_dir)
    command.append('%d' %batch_size)
    command.append(common_udpipe_future.get_training_schedule(epochs))
    for i in range(2):
        if len(monitoring_datasets) > i:
            command.append(monitoring_datasets[i].filename)
    print('Running', command)
    sys.stderr.flush()
    sys.stdout.flush()
    subprocess.call(command)
    if common_udpipe_future.incomplete(model_dir):
        if common_udpipe_future.memory_error(model_dir):
            # do not leave erroneous model behind
            os.rename(model_dir, model_dir+('-oom-%d' %batch_size))
            # try again with smaller batch size:
            if batch_size == 1:
                raise ValueError('Cannot train parser even with batch size 1.')
            new_batch_size = int(batch_size/2)
            print('Parser ran out of memory. Re-trying with batch size %d' %new_batch_size)
            train(dataset_filename, seed, model_dir,
                epoch_selection_dataset = epoch_selection_dataset,
                monitoring_datasets = monitoring_datasets,
                lcode = lcode,
                batch_size = new_batch_size,
            )
        else:
            # do not leave incomplete model behind
            error_name = model_dir + '-incomplete'
            os.rename(model_dir, error_name)
            raise ValueError('Model is missing essential files: ' + error_name)

def predict(model_path, input_path, prediction_output_path):
    command = []
    command.append('./elmo_udpf-predict.sh')
    command.append(model_path)
    command.append(input_path)
    command.append(prediction_output_path)
    print('Running', command)
    sys.stderr.flush()
    sys.stdout.flush()
    subprocess.call(command)

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()

