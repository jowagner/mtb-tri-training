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

import hashlib
import os
import subprocess
import sys
import time

def print_usage():
    print('Usage: %s [options]' %(os.path.split(sys.argv[0])[-1]))
    print("""
Options:

    --deadline  HOURS       Do not train another model after HOURS hours
                            and quit script.
                            (Default: 0.0 = no limit)

    --stopfile  FILE        Do not train another model if FILE exists.

""")

def get_training_schedule(epochs = 60):
    ''' return a udpipe-future learning rate schedule
        and epochs specification like
        "30:1e-3,5:6e-4,5:4e-4,5:3e-4,5:2e-4,10:1e-4"
        adjusted to the given number of epochs
    '''
    if type(epochs) is str:
        epochs = int(epochs)
    if 'TT_DEBUG' in os.environ \
    and os.environ['TT_DEBUG'].lower() not in ('0', 'false'):
        epochs = 1 + int(epochs/20)
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

def my_makedirs(required_dir):
    if not os.path.exists(required_dir):
        try:
            os.makedirs(required_dir)
        except OSError:
            # folder was created by another process
            # between the to calls above
            # (in python 3, we will be able to use exist_ok=True)
            pass

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
        task_dir  = os.environ['TT_TASK_DIR']
        inbox_dir = task_dir + '/udpf/inbox'
        filename = '%s/%s-%d.task' %(
            inbox_dir,
            task_id,
            my_task_bucket,
        )
        if 'TT_TASK_PATIENCE' in os.environ:
            patience = float(os.environ['TT_TASK_PATIENCE'])
        else:
            patience = 36000.0
        expires = now + patience
        my_makedirs(inbox_dir)
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
        if 'TT_DEBUG' in os.environ:
            verbosity_interval /= 100.0
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
            task_dir,
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
            if 'TT_DEBUG' in os.environ:
                verbosity_interval /= 100.0
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
            task_dir,
            my_task_bucket,
            task_id,
        )
        if not os.path.exists(filename):
            print('Task %s failed' %task_id)
            raise ValueError('Task %s marked by task master as no longer active but not as complete')
        if 'TT_TASK_ARCHIVE_COMPLETED' in os.environ  and \
        os.environ['TT_TASK_ARCHIVE_COMPLETED'].lower() not in ('0', 'false'):
            archive_dir = '%s/udpf/archive' %task_dir
            my_makedirs(archive_dir)
            counter = 1
            while True:
                archive_name = '%s/%s-run%03d.task' %(archive_dir, task_id, counter)
                if os.path.exists(archive_name):
                    counter += 1
                    continue
                os.rename(filename, archive_name)
                break
        elif 'TT_TASK_CLEANUP_COMPLETED' in os.environ  and \
        os.environ['TT_TASK_CLEANUP_COMPLETED'].lower() not in ('0', 'false'):
            os.unlink(filename)

def worker():
    opt_help = False
    opt_debug = True
    opt_deadline = None
    opt_stopfile = None
    while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
        option = sys.argv[1]
        option = option.replace('_', '-')
        del sys.argv[1]
        if option in ('--help', '-h'):
            opt_help = True
            break
        elif option == '--deadline':
            opt_deadline = 3600.0 * float(sys.argv[1])
            if opt_deadline:
                opt_deadline += time.time()
            del sys.argv[1]
        elif option == '--stopfile':
            opt_stopfile = sys.argv[1]
            del sys.argv[1]
        else:
            print('Unsupported or not yet implemented option %s' %option)
            opt_help = True
            break
    if len(sys.argv) != 1:
        opt_help = True
    if opt_help:
        print_usage()
        sys.exit(0)
    tt_task_dir = os.environ['TT_TASK_DIR']
    inbox_dir  = tt_task_dir + '/udpf/inbox'
    active_dir = tt_task_dir + '/udpf/active'
    final_dir  = tt_task_dir + '/udpf/completed'
    for required_dir in (inbox_dir, active_dir, final_dir):
        my_makedirs(required_dir)
    while True:
        if opt_deadline and time.time() > opt_deadline:
            print('\n*** Reached deadline. ***\n')
            sys.exit(0)
        if opt_stopfile and os.path.exists(opt_stopfile):
            print('\n*** Found stop file. ***\n')
            sys.exit(0)
        candidate_tasks = []
        for filename in os.listdir(inbox_dir):
            if filename.endswith('.task') and '-' in filename:
                candidate_tasks.append(filename)
        candidate_tasks.sort()
        for filename in candidate_tasks:
            taskfile    = '%s/%s' %(inbox_dir, filename)
            task_id, task_bucket = filename[:-5].rsplit('-', 1)
            bucket_dir  = '%s/%s' %(active_dir, task_bucket)
            my_makedirs(bucket_dir)
            active_name = '%s/%s.task' %(bucket_dir, task_id)
            try:
                os.rename(taskfile, active_name)
            except:
                if opt_debug:
                    print('Task %s claimed by other worker' %task_id)
                continue
            f = open(active_name, 'rb')
            exp_line = f.readline()
            fields = exp_line.split()
            if not exp_line.startswith('expires') \
            or len(fields) != 2:
                print('Deleting malformed task', task_id)
                f.close()
                os.unlink(active_name)
                continue
            expires = float(fields[1])
            if time.time() > expires:
                print('Deleting expired task', task_id)
                f.close()
                os.unlink(active_name)
            command = f.read().split('\n')
            f.close()
            # handle last line with linebreak
            if command and command[-1] == '':
                del command[-1]
            # found the first task eligible to run
            submit_time = os.path.getmtime(active_name)
            start_time = time.time()
            print('Running task %s: %r' %(task_id, command))
            sys.stderr.flush()
            sys.stdout.flush()
            subprocess.call(command)
            end_time = time.time()
            # signal completion
            bucket_dir = '%s/%s' %(final_dir, task_bucket)
            my_makedirs(bucket_dir)
            final_file = '%s/%s.task' %(bucket_dir, task_id)
            f = open(final_file, 'wb')
            f.write('duration\t%.1f\n' %(end_time-start_time))
            f.write('waiting\t%.1f\n' %(start_time-submit_time))
            f.write('total\t%.1f\n' %(end_time-submit_time))
            f.write('cluster\t%s\n' %(os.environ['SLURM_CLUSTER_NAME']))
            f.write('job_id\t%s\n' %(os.environ['SLURM_JOB_ID']))
            f.write('job_name\t%s\n' %(os.environ['SLURM_JOB_NAME']))
            f.write('host\t%s\n' %(os.environ['HOSTNAME']))
            f.write('process\t%d\n' %os.getpid())
            f.write('submitted\t%.1f\n' %submit_time)
            f.write('start\t%.1f\n' %start_time)
            f.write('end\t%.1f\n' %end_time)
            f.write('expires\t%.1f\n' %expires)
            f.write('task_id\t%s\n' %task_id)
            f.write('bucket\t%s\n' %task_bucket)
            f.write('arg_len\t%d\n' %len(command))
            f.write('\n') # empty line to mark end of header, like in http
            f.write('\n'.join(command))
            f.close()
            os.unlink(active_name)
            # before processing any remaining task, let the
            # outer loop refresh the task list
            break
        time.sleep(0.25)  # poll interval while queue is empty

if __name__ == "__main__":
    worker()

