#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import collections
import os

def train(dataset_filename, seed, model_dir, epoch_selection_dataset = None):
    if epoch_selection_dataset:
        raise ValueError('Epoch selection not supported with udpipe-future.')
    command = []
    command.append('./udpipe-future-train.sh')
    command.append(dataset_filename)
    command.append(seed)
    command.append(model_dir)
    sys.stderr.flush()
    sys.stdout.flush()
    subprocess.call(command)

def main():
    import sys
    raise NotImplementedError

if __name__ == "__main__":
    main()

