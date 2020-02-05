#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

import os
import sys
import time

"""
Usage: grep -HE "(Score|Iteration|Subtotal|[mM]odel path)" */stdout.txt | ./log-to-scores.py > scores.tsv
or   : grep -HE "(Score|Iteration|Subtotal|[mM]odel path)" */stdout.txt | ./log-to-scores.py --update scores.tsv
"""

def get_date(path):
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        mtime_struct = time.localtime(mtime)
        y_m_d = mtime_struct[:3]
        return '%04d-%02d-%02d' %y_m_d
    else:
        return '????-??-??'

key2scores = {}
key2tokens = {}
key2dates = {}
last_filename = None
max_rounds = 0
while True:
    line = sys.stdin.readline()
    if not line:
        break
    fields = line.replace(':', ' ').split()
    filename = fields[0]
    if filename != last_filename:
        code = filename.split('/')[-2]
        lang = code[0]
        parser = code[1]
        method = code[2:6]
        num_learners = int(code[6])
        augment_size_index = int(code[7])
        last_filename = filename
        tt_round = 0
        score_index = 0
        test_set_index = 0
        learner_unlabelled_tokens = []
        learner_unlabelled_sentences = []
        learner_model_time = []
    if '== Tri-training Iteration' in line:
        if fields[-1].endswith(']'):
            # remove [timestamp]
            while True:
                last_item = fields[-1]
                del fields[-1]
                if last_item.startswith('['):
                    break
        tt_round += 1
        if fields[-4] != '%d' %tt_round:
            raise ValueError('logfile %s contains data for round %s where data for round %d is expected' %(
                filename, fields[-4], tt_round
            ))
        score_index = 0
        test_set_index = 0
        learner_unlabelled_tokens = []
        learner_unlabelled_sentences = []
        learner_model_time = []
    elif 'Subtotal:' in line:
        # [0]      [1]       [2]   [3]   [4] [5] [6]
        # filename Subtotal  75323 items in 7426 sentences.
        if len(fields) != 7 or fields[3] != 'items' or fields[6] != 'sentences.':
            continue
        learner_unlabelled_tokens.append(int(fields[2]))
        learner_unlabelled_sentences.append(int(fields[5]))
    elif 'Score:' in line:
        if tt_round > max_rounds:
            max_rounds = tt_round
        score = fields[-1]
        if score_index < num_learners:
            learner = 'L%d' %(score_index+1)
        else:
            learner = 'Ensemble'
        key = (
            lang, parser,
            '%d' %augment_size_index,
            method, '%d' %num_learners,
            learner, '%d' %test_set_index
        )
        if not key in key2scores:
            if tt_round:
                raise ValueError('New key %r but no data for round 0' %(key,))
            key2scores[key] = []
            key2tokens[key] = []
            key2dates[key] = []
            # round 0 has no unlabelled data
            key2tokens[key].append((0,0))
        else:
            if learner == 'Ensemble':
                n_tokens = sum(learner_unlabelled_tokens)
                n_sentences = sum(learner_unlabelled_sentences)
            else:
                n_tokens = learner_unlabelled_tokens[score_index]
                n_sentences = learner_unlabelled_sentences[score_index]
            key2tokens[key].append((n_tokens, n_sentences))
        if learner == 'Ensemble':
            date = max(learner_model_time)
        else:
            date = learner_model_time[score_index]
        key2dates[key].append(date)
        key2scores[key].append(score)
        if score_index == num_learners:
            score_index = 0
            test_set_index += 1
        else:
            score_index += 1
    elif 'Model path:' in line:
        model_dir = fields[-1]
        date = get_date(model_dir + '/training.end')
        learner_model_time.append(date)
    elif 'Adjusting model path to existing model' in line:
        model_dir = fields[-1].strip("'")
        date = get_date(model_dir + '/training.end')
        learner_model_time[-1] = date

key_and_rounds_header = '\t'.join([
    'Language', 'Parser',
    'AugmentSizeIndex',
    'Method', 'NumberOfLearners',
    'Learner', 'TestSetIndex', 'Rounds',
    'UnlabelledTokensInLastRound',
])

backup_stdout = sys.stdout
if len(sys.argv) > 1:
    if sys.argv[1] != '--update':
        raise ValueError('unknown option %r' %(sys.argv[1]))
    filename = sys.argv[2]
    if os.path.exists(filename):
        f = open(filename, 'rb')
        old_header = f.readline()
        scores_start = 9
        if not old_header.startswith(key_and_rounds_header):
            scores_start = 8
            #raise ValueError('Unsupported tsv format')
        while True:
            line = f.readline()
            if not line:
                break
            fields = line.split()
            key = tuple(fields[:7])
            scores = fields[scores_start:]
            if scores_start == 8:
                n_tokens = -1
            else:
                n_tokens = int(fields[8])
            if key not in key2scores \
            or len(scores) > len(key2scores[key]):
                key2scores[key] = scores
                key2tokens[key] = ((len(scores)-1) * [(-1, -1)]) + [(n_tokens, -1)]
        f.close()
    sys.stdout = open(filename, 'wb')

# table header
sys.stdout.write(key_and_rounds_header)
for i in range(max_rounds+1):
    sys.stdout.write('\t%d' %i)
sys.stdout.write('\n')
# table body
for key in sorted(key2scores):
    sys.stdout.write('\t'.join(key))
    scores = key2scores[key]
    sys.stdout.write('\t%d\t' %(len(scores)-1))
    sys.stdout.write('%d' %(key2tokens[key][-1][0]))
    token_info = key2tokens[key]
    dates = key2dates[key]
    for index, score in enumerate(scores):
        sys.stdout.write('\t')
        sys.stdout.write('%s:%s:%d:%d' %(
            score, dates[index], tokens[index], sentences[index]
        ))
    sys.stdout.write('\n')

if sys.stdout != backup_stdout:
    sys.stdout.close()
    sys.stdout = backup_stdout

