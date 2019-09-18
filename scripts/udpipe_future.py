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

import subprocess
import sys

def train(
    dataset_filename, seed, model_dir,
    epoch_selection_dataset = None,
    monitoring_datasets = []
):
    if epoch_selection_dataset:
        raise ValueError('Epoch selection not supported with udpipe-future.')
    command = []
    command.append('./udpipe_future-train.sh')
    command.append(dataset_filename)
    command.append(seed)
    command.append(model_dir)
    for i in range(2):
        if len(monitoring_datasets) > i:
            command.append(monitoring_datasets[i].filename)
    print('Running', command)
    sys.stderr.flush()
    sys.stdout.flush()
    subprocess.call(command)

def predict(model_path, input_path, prediction_output_path):
    command = []
    command.append('./udpipe_future-predict.sh')
    command.append(model_path)
    command.append(input_path)
    command.append(prediction_output_path)
    print('Running', command)
    sys.stderr.flush()
    sys.stdout.flush()
    subprocess.call(command)

def evaluate(prediction_path, gold_path):
    command = []
    command.append('./conll18_ud_eval.py')
    command.append('--verbose')
    command.append(gold_path)
    command.append(prediction_path)
    print('Running', command)
    sys.stderr.flush()
    sys.stdout.flush()
    subprocess.call(command)

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()

