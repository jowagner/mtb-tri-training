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
import subprocess
import sys
import time
from collections import defaultdict

def main():
    opt_script_dir = '/'.join((os.environ['PRJ_DIR'], 'scripts'))
    opt_stopfile = 'stop-fill-queue'
    opt_stop_check_interval = 12.0
    opt_submit_interval = 3600.0
    opt_username = getpass.getuser()
    start_time = time.time()
    earliest_next_submit = start_time
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
        output = subprocess.check_output(command) # an exception is raised if the command fails
        queue = defaultdict(lambda: 0)
        for row in output.split('\n'):
            row.rstrip()
            if not row:
                continue
            fields = row.split()
            # light check for expected format
            assert len(fields) == 8
            assert fields[3] == opt_username
            # extract data
            job_name = fields[2]
            job_state = fields[4]
            queue[(job_name, job_state)] += 1
        print('My jobs at', time.ctime(now))
        for key in queue:
            print('\t%r with frequency %d' %(key, queue[key]))
        # check what to submit
        for job_name, script_name, max_waiting, max_running in [
            ('udpfr30h', 'worker-udpf-grove-rtx.job',      0, 10),
            ('udpft30h', 'worker-udpf-grove-tesla.job',    0,  4),
            ('udpf430h', 'worker-udpf-grove-titanv.job',   0,  1),
            ('e5wo-30h', 'worker-elmo-hdf5-grove-cpu.job', 0,  8),
        ]:
            if queue[(job_name, 'PD')] > max_waiting:
                continue
            if queue[(job_name, 'R')] > max_running:
                continue
            # submit job
            command = ['sbatch', '/'.join((opt_script_dir, script_name))]
            subprocess.call(command)
            print('Submitted %s (%s)' %(job_name, script_name))
            # move forward time for next job submission
            now = time.time()
            while earliest_next_submit <= now:
                earliest_next_submit += opt_submit_interval
            # only submit 1 job per occasion
            break

if __name__ == "__main__":
    main()

