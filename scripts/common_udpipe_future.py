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
import time

def get_training_schedule(epochs = 60):
    ''' return a udpipe-future learning rate schedule
        and epochs specification like
        "30:1e-3,5:6e-4,5:4e-4,5:3e-4,5:2e-4,10:1e-4"
        adjusted to the given number of epochs
    '''
    if type(epochs) is str:
        epochs = int(epochs)
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

def run_command(command):
    if 'TT_TASK_DIR' not in os.environ:
        print('Running', command)
        sys.stderr.flush()
        sys.stdout.flush()
        subprocess.call(command)
    else:
        # prepare task submission file
        now = time.time()
        if 'TT_TASK_EPOCH' in os.environ:
            t0 = float(os.environ['TT_TASK_EPOCH'])
        else:
            t0 = 0.0
        command_fingerprint = hashlib.sha256('\n'.join(command)).hexdigest()
        task_id = '%05x-%s-%s-%d-%s-%s' %(
            int((now-t0)/60.0),
            os.environ['HOSTNAME'].replace('-', '_'),
            os.environ['SLURM_JOB_ID'],
            os.getpid(),
            command_fingerprint[:8],
            command_fingerprint[8:16],
        )
        num_task_buckets = int(os.environ['TT_TASK_BUCKETS'])
        task_fingerprint = hashlib.sha512(task_id).hexdigest()
        my_task_bucket = int(task_fingerprint, 16) % num_task_buckets
        filename = '%s/udpf/inbox/%s-%d.task' %(
            os.environ['TT_TASK_DIR'],
            task_id,
            my_task_bucket,
        )
        if 'TT_TASK_PATIENCE' in os.environ:
            patience = float(os.environ['TT_TASK_PATIENCE'])
        else:
            patience = 36000.0
        expires = now + patience
        f = open(filename+'.prep', 'wb')
        f.write('expires %.1f\n' %expires)
        f.write('\n'.join(command))
        f.close()
        os.rename(filename+'.prep', filename)
        submit_time = time.time()
        print('Submitted task %s with command %r' %(task_id, command))
        sys.stderr.flush()
        sys.stdout.flush()
        # wait for task to start
        iteration = 0
        fp_length = len(task_fingerprint)
        verbosity_interval = 3600.0
        next_verbose = submit_time + verbosity_interval
        start_time_interval = (submit_time, submit_time)
        while time.time() < expires and os.path.exists(filename):
            duration = 30.0 + int(task_fingerprint[iteration % fp_length], 16)
            now = time.time()
            start_time_interval = (now, now+duration)
            time.sleep(duration)
            iteration += 1
            now = time.time()
            if now >= next_verbose:
                print('Waited %.1f hours so far for task %s to start' %(
                    (now-submit_time)/3600.0,
                    task_id,
                ))
                verbosity_interval *= 1.4
                next_verbose += verbosity_interval
        # did it expire?
        has_expired = True
        # check for file in active queue
        time.sleep(5.0) # just in case moving the file is not atomic
        filename = '%s/udpf/active/%d/%s.task' %(
            os.environ['TT_TASK_DIR'],
            my_task_bucket,
            task_id,
        )
        if os.path.exists(filename):
            has_expired = False
            # expectation value of start time assuming uniform
            # distribution within above sleep interval
            start_time = sum(start_time_interval) / 2.0
            # wait for task to finish (no timeout)
            verbosity_interval = 3600.0
            next_verbose = min(next_verbose, start_time + verbosity_interval)
            while os.path.exists(filename):
                duration = 30.0 + int(task_fingerprint[iteration % fp_length], 16)
                now = time.time()
                end_time_interval = (now, now+duration)
                time.sleep(duration)
                iteration += 1
                now = time.time()
                if now >= next_verbose:
                    print('Task %s is running about %.1f hours so far' %(
                        task_id,
                        (now-start_time)/3600.0,
                    ))
                    verbosity_interval *= 1.4
                    next_verbose += verbosity_interval
        # task is not active --> check for completion
        filename = '%s/udpf/completed/%d/%s.task' %(
            os.environ['TT_TASK_DIR'],
            my_task_bucket,
            task_id,
        )
        if not os.path.exists(filename):
            print('Task %s failed' %task_id)
            raise ValueError('Task %s marked by task master as no longer active but not as complete')

def task_master():
    # wait for task to appear
    # extract bucket and task_id
    # move file to active folder
    # run task
    # append stats to active file
    # move active file to completed folder

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()

