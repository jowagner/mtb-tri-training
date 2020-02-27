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
    batch_size = 32,
    epochs = 60,
    priority = 50,
    is_multi_treebank = False,
    submit_and_return = False,
):
    if epoch_selection_dataset:
        raise ValueError('Epoch selection not supported with udpipe-future.')
    if is_multi_treebank:
        raise ValueError('Multi-treebank models not supported by current wrapper script.')
    command = []
    command.append('./udpipe_future-train.sh')
    command.append(dataset_filename)
    if seed is None:
        raise NotImplementedError
    command.append(seed)
    command.append(model_dir)
    command.append('%d' %batch_size)
    command.append(common_udpipe_future.get_training_schedule(epochs))
    for i in range(2):
        if len(monitoring_datasets) > i:
            command.append(monitoring_datasets[i].filename)
    task = common_udpipe_future.run_command(
        command,
        priority = priority,
        submit_and_return = submit_and_return,
    )
    if submit_and_return:
        return task
    check_model(model_dir)

def check_model(model_dir):
    if not os.path.exists(model_dir):
        raise ValueError('Failed to train parser (missing output)')
    if common_udpipe_future.incomplete(model_dir):
        # do not leave erroneous model behind
        if common_udpipe_future.memory_error(model_dir):
            # out-of-memory error detected
            error_name = model_dir + '-oom'
        else:
            # other errors
            error_name = model_dir + '-incomplete'
        os.rename(model_dir, error_name)
        raise ValueError('Model is missing essential files: ' + error_name)

def predict(
    model_path, input_path, prediction_output_path,
    priority = 50,
    is_multi_treebank = False,
    submit_and_return = False,
    wait_for_input = False,
    wait_for_model = False,
):
    if is_multi_treebank:
        raise ValueError('Multi-treebank models not supported by current wrapper script.')
    command = []
    command.append('./udpipe_future-predict.sh')
    command.append(model_path)
    command.append(input_path)
    command.append(prediction_output_path)
    requires = []
    if wait_for_input:
        requires.append(input_path)
    if wait_for_model:
        requires.append(model_path)
    task = common_udpipe_future.run_command(
        command,
        requires = requires,
        priority = priority,
        submit_and_return = submit_and_return,
    )
    if submit_and_return:
        return task

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()

