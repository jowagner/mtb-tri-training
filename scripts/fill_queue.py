#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

from __future__ import print_function

import getpass
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

def get_resource(jobs, job_name):
    for prefix, resource, _, _, _ in jobs:
        if job_name.startswith(prefix):
            return resource
    # master and cache jobs
    return None

def main():
    opt_jobs = [
        # job_name_prefix, resource, script_name, max_waiting, max_running
        ('udpfr', 'rtx',    'worker-udpf-grove-rtx.job',          0,  2),
        ('udpft', 'tesla',  'worker-udpf-grove-tesla.job',        0,  2),
        ('udpfv', 'titanv', 'worker-udpf-grove-titanv.job',       0,  1),
        ('udpfq', 'quadro', 'worker-udpf-grove-quadro.job',       0,  2),
        ('e5wo-', 'no-gpu', 'worker-elmo-hdf5-grove-cpu.job',     0,  4),
        ('m5wo-', 'no-gpu', 'worker-mbert-hdf5-grove-cpu.job',    0,  4),
        ('m5wq-', 'quadro', 'worker-mbert-hdf5-grove-quadro.job', 0,  1),
        ('m5wr-', 'rtx',    'worker-mbert-hdf5-grove-rtx.job',    0,  1),
        ('m5wv-', 'titanv', 'worker-mbert-hdf5-grove-titanv.job', 0,  1),
    ]
    opt_max_submit_per_occasion = 1
    opt_script_dir = '/'.join((os.environ['PRJ_DIR'], 'scripts'))
    opt_task_dir   = os.environ['TT_TASK_DIR']
    opt_stopfile = 'stop-fill-queue'
    opt_stop_check_interval = 12.0
    opt_submit_interval = 1800.0
    opt_probe_interval  = 300.0
    opt_username = getpass.getuser()
    start_time = time.time()
    earliest_next_submit = start_time
    has_tasks = {}
    while True:
        exit_reason = None
        if opt_stopfile and os.path.exists(opt_stopfile):
            exit_reason = 'Found stop file'
        if exit_reason:
            print('\n*** %s ***\n' %exit_reason)
            sys.exit(0)
        now = time.time()
        if now < earliest_next_submit:
            # wait no more than `opt_stop_check_interval` seconds so that
            # stop conditions are checked regularly and avoid leaving a
            # very short waiting period to the end
            wait = min(2*opt_stop_check_interval, earliest_next_submit - now)
            if wait > opt_stop_check_interval:
                wait = wait / 2
            time.sleep(wait)
            continue
        # get queue state
        command = ['squeue', '--noheader', '--user=' + opt_username]
        try:
            output = subprocess.check_output(command) # an exception is raised if the command fails
        except subprocess.CalledProcessError:
            print('Error checking job queue, trying again in a minute')
            earliest_next_submit = time.time() + 60.0
            continue
        queue = defaultdict(lambda: 0)
        for row in output.split('\n'):
            row.rstrip()
            if not row:
                continue
            fields = row.split()
            # light check for expected format
            assert len(fields) >= 8
            assert fields[3] == opt_username
            # extract data
            job_name = fields[2][:-3]  # remove suffix with estimated duration
            job_state = fields[4]
            queue[(job_name, job_state)] += 1
            resource = get_resource(opt_jobs, job_name)
            queue[(resource, job_state)] += 1
        print('My jobs at', time.ctime(now))
        for key in sorted(list(queue.keys())):
            print('\t%r with frequency %d' %(key, queue[key]))
        # check what may be needed
        for inbox in ('udpf', 'elmo-hdf5', 'mbert-hdf5'):
            inbox_path = '/'.join((opt_task_dir, inbox, 'inbox'))
            has_tasks[inbox] = False
            for inbox_f in os.listdir(inbox_path):
                if inbox_f.endswith('.task'):
                    has_tasks[inbox] = True
                    break
        # check what to submit
        std_jobs = []
        prio_jobs = []
        secondary_prio_jobs = []
        for job_item in opt_jobs:
            job_name = job_item[0]
            if job_name.startswith('e5w') and has_tasks['elmo-hdf5'] \
            or job_name.startswith('m5w') and has_tasks['mbert-hdf5']:
                resource = job_item[1]
                if resource == 'no-gpu':
                    # slow mbert or elmo job --> not preferred
                    secondary_prio_jobs.append(job_item)
                else:
                    prio_jobs.append(job_item)
                print('Prioritising', job_name)
            else:
                std_jobs.append(job_item)
        random.shuffle(prio_jobs)
        random.shuffle(secondary_prio_jobs)
        random.shuffle(std_jobs)
        opt_jobs = prio_jobs + secondary_prio_jobs + std_jobs
        n_submitted = 0
        for job_name, resource, script_name, max_waiting, max_running in opt_jobs:
            if job_name.startswith('e5w') and not has_tasks['elmo-hdf5']:
                continue
            if job_name.startswith('m5w') and not has_tasks['mbert-hdf5']:
                continue
            if job_name.startswith('udpf') and not has_tasks['udpf']:
                continue
            if queue[(resource, 'PD')] > max_waiting:
                continue
            if queue[(job_name, 'PD')] > max_waiting:
                continue
            if queue[(job_name, 'R')] > max_running:
                continue
            # submit job
            sys.stdout.flush()   # make sure command output appears after our last output
            command = ['sbatch', '/'.join((opt_script_dir, script_name))]
            try:
                subprocess.call(command)
            except subprocess.CalledProcessError:
                print('Error submitting job, trying again in a minute')
                earliest_next_submit = time.time() + 60.0
                break
            print('Submitted %s (%s)' %(job_name, script_name))
            # move forward time for next job submission
            now = time.time()
            while earliest_next_submit <= now:
                earliest_next_submit += opt_submit_interval
            # limit how many jobs to submit at each occasion
            n_submitted += 1
            if n_submitted >= opt_max_submit_per_occasion:
                break
        now = time.time()
        while earliest_next_submit <= now:
            earliest_next_submit += opt_probe_interval
        sys.stdout.flush()

if __name__ == "__main__":
    main()

