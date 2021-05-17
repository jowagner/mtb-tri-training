#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

# TODO: move code shared with computation-budget-plot-for-baseline.py
#       into a module

from __future__ import print_function

import hashlib
import os
import pickle
import random
import sys
import time

import basic_dataset
import conllu_dataset

start_t = time.time()

input_dir = os.environ['PRJ_DIR'] + '/workdirs'

if sys.argv[1] == '--show-schedule':
    opt_show_schedule = True
    del sys.argv[1]
else:
    opt_show_schedule = False

language = sys.argv[1]
number_of_bins = int(sys.argv[2])
number_of_samples = int(sys.argv[3])
random.seed(int(hashlib.sha256(sys.argv[4]).hexdigest(), 16))
opt_augexp = sys.argv[5]

l2text = {
    'e': 'English',
    'h': 'Hungarian',
    'u': 'Uyghur',
    'v': 'Vietnamese',
}
p2text = {
    'f': 'udpipe-future',
    'g': '+fasttext',
    'h': '+elmo',
    'i': 'udpf+fasttext+mBERT',
}
s2text = {
    '-': 'bootstrap samples of labelled data (100% of size of data)',
    #'p': 'bootstrap samples of labelled data (300% of size of data)',
    #'t': '300% of labelled data, i.e. concatenation of 3 copies',
    'w': 'permutations of labelled data, i.e. shuffling the data',
    'x': '250% of labelled data, i.e. concatenation of 2 1/2 copies',
}

if language in ('*', 'all'):
    languages = sorted(list(l2text.keys()))
    print('All languanges selected:', languages)
else:
    print('Language:', l2text[language])
    languages = [language]

parsers = list(p2text.keys())

def get_training_duration(path):
    try:
        start_t = os.path.getmtime('%s/training.start' %path)
    except:
        start_t = None
    try:
        end_t = os.path.getmtime('%s/training.end' %path)
    except:
        end_t = None
    if start_t and end_t:
        return end_t - start_t
    else:
        print('Warning: Could not get duration of', path)
    return None

def get_number_of_tokens(path):
    dataset = basic_dataset.load_or_map_from_filename(
        conllu_dataset.new_empty_set(),
        path
    )
    n_tokens = 0
    for sentence in dataset:
        n_tokens += len(sentence)
    return n_tokens

def estimate_ext_embedding_duration(path, parser):
    n_tokens = get_number_of_tokens(path)
    duration = 0.0
    if parser == 'i':  # mBERT
        duration = 50.0 * n_tokens / 15000.0
    if parser == 'h':  # ELMo
        duration = 25.0 * n_tokens / 15000.0   # TODO: measure speed
    if parser != 'f':
        duration += 10.0  # constant loading time for fasttext
    return duration

def estimate_prediction_duration(path, parser):
    duration = estimate_ext_embedding_duration(path, parser)
    duration += 30.0   # estimate for running the parser (mostly model loading)
    return duration

def iteration_to_int(iteration):
    if iteration[0] == '0':
        iteration = iteration[1:]
    return int(iteration)

def fix_data_bugs(data):
    move = []
    for key in data:
        if len(key) == 2  \
        and key[1].startswith('nt-duration-')  \
        and '.' in key[1]:
            new_key = (key[0], key[1].split('.')[0])
            move.append((key, new_key))
    for key, new_key in move:
        data[new_key] = data[key]
        del data[key]
    return data

def get_run_data(run_dir, parser):
    cache_filename = '%s/run-info-cache.pickle' %run_dir
    if os.path.exists(cache_filename):
        with open(cache_filename, 'r') as f:
            data = pickle.load(f)
            data = fix_data_bugs(data)
        data['run_dir'] = run_dir
        return data
    data = {}
    for entry in os.listdir(run_dir):
        if entry.endswith('workdir') \
        or '-incomplete-' in entry:
            continue
        path = '/'.join([run_dir, entry])
        fields = entry.replace('.', '-').split('-')
        if entry.startswith('model-'):
            iteration    = iteration_to_int(fields[1])
            learner_rank = fields[2]
            duration     = get_training_duration(path)
            if duration:
                data[(iteration, 'tr-duration-'+learner_rank)] = duration
        elif entry.startswith('new-training-set-') \
        and '.conllu' in entry:
            iteration    = iteration_to_int(fields[3])
            learner_rank = fields[4]
            duration = estimate_ext_embedding_duration(path, parser)
            if duration:
                data[(iteration, 'nt-duration-'+learner_rank)] = duration
        elif entry.startswith('prediction-') \
        and '.conllu' in entry:
            if '-test-' in entry:
                continue
            learner_rank = fields[2]
            if learner_rank == 'E':
                continue
            iteration = iteration_to_int(fields[1])
            duration  = estimate_prediction_duration(path, parser)
            if duration:
                if '-dev-' in entry:
                    data[(iteration, 'pr-duration-'+learner_rank)] = duration
                elif '-subset-part-001-' in entry \
                or ('-subset-' in entry and not '-part-' in entry):
                    data[(iteration, 'sp-duration-'+learner_rank)] = duration
                else:
                     raise ValueError('unexpected file %r in %r' %(entry, run_dir))
        elif entry.startswith('prediction-') \
        and entry.endswith('.eval.txt'):
            if '-test-' in entry or not '-dev-' in entry:
                continue
            score = conllu_dataset.get_score_from_file(path)[0]
            s_time = os.path.getmtime(path)
            score = (s_time, score)
            learner_rank = fields[2]
            key = (iteration, 'score-'+learner_rank)
            if key in data and data[key] > score:  # use the newest file
                score = data[key]
            data[key] = score
        elif entry.startswith('seed-set-') \
        and '.conllu' in entry:
            iteration    = 0
            learner_rank = fields[2]
            duration = estimate_ext_embedding_duration(path, parser)
            if duration:
                data[(iteration, 'nt-duration-'+learner_rank)] = duration
    with open(cache_filename, 'w') as f:
        pickle.dump(data, f)
    data['run_dir'] = run_dir
    return data

def get_iteration(data, parser, iteration):
    list_of_tables = []
    if iteration > 0:
        d = 0.0
        for learner in '123':
            d += data[(iteration, 'sp-duration-'+learner)]  # subset prediction (unlabelled data)
        d += 30.0   # filtering data for agreement etc.
        list_of_tables.append([(d, 0.0)])
    table = []
    for learner in '123':
        d = 0.0
        d += data[(iteration, 'nt-duration-'+learner)]  # embeddings for new training data
        d += data[(iteration, 'tr-duration-'+learner)]  # training the model
        d += data[(iteration, 'pr-duration-'+learner)]  # predict and eval
        _, score = data[(iteration, 'score-'+learner)]
        table.append((d, score))
    list_of_tables.append(table)
    # ensemble
    _, score = data[(iteration, 'score-E')]
    list_of_tables.append([(21*0.3, score)])
    return list_of_tables

def get_run(run_dir, parser):
    print('run_dir', run_dir)
    data = get_run_data(run_dir, parser)
    run = []
    t = 0
    while True:
        iteration = get_iteration(data, parser, t)
        #try:
        #    iteration = get_iteration(data, parser, t)
        #except KeyError:
        #    break
        run.append(iteration)
        t += 1
    return run

print('Scanning folders:')
available_runs = []
candidates = []
for entry in os.listdir(input_dir):
    if len(entry) != 8:
        continue
    language = entry[0]
    parser   = entry[1]
    sampling = entry[3]
    augexp   = entry[7]
    if augexp != opt_augexp:
        continue
    if language not in languages:
        continue
    if parser not in parsers:
        continue
    #if augexp not in '68A':
    #    continue
    candidates.append((
        '/'.join([input_dir, entry]),
        parser,
    ))

n_candidates = len(candidates)
last_verbose = 0.0
for c_index, candidate in enumerate(candidates):
    candidate_folder, parser = candidate
    now = time.time()
    if now > last_verbose + 1.0:
        percentage = 100.0 * c_index / float(n_candidates)
        print('%.1f%% done' %percentage)
        last_verbose = now
    run = get_run(candidate_folder, parser)
    if run:
        available_runs.append((language, run))
print('Finished')

# rename to keep name from other script below
available_ensembles = available_runs

def add_ensemble_to_budget(budget, row, language):
    duration = 0.3      # estimate for combiner and eval runs (10x parallel)
    score = float(row[0])
    budget.append((language, duration, score))

def bring_eligble_ensembles_forward(
    start, language, completed_ensembles, available_ensembles, budget,
    predictions_made, candidate_rows, new_prediction
):
    for row_index in candidate_rows[new_prediction]:
        if row_index < start \
        or (language, row_index) in completed_ensembles:
            continue
        row = available_ensembles[row_index]
        # are all neccessary predictions ready?
        ready = True
        for index in [14, 15, 16]:
             prediction = row[index]
             if not prediction in predictions_made:
                 ready = False
                 break
        if ready:
            # all predictions are ready
            add_ensemble_to_budget(budget, row, language)
            completed_ensembles.add((language, row_index))

def get_candidate_rows(lang2available_ensembles):
    retval = {}
    for language in lang2available_ensembles:
        available_ensembles = lang2available_ensembles[language]
        for row_index, row in enumerate(available_ensembles):
            for index in [14, 15, 16]:
                prediction = row[index]
                if prediction not in retval:
                    retval[prediction] = []
                retval[prediction].append(row_index)
    return retval

def enumerate_rows(run, language):
    for iteration in run:
        for table in iteration:
            if len(table) > 1:
                random.shuffle(table)
            for (duration, score) in table:
                yield (language, duration, score)

def expand_steps(available_ensembles, languages):
    retval = []
    # re-organise languages in separate streams
    lang2available_ensembles = {}
    for language in languages:
        lang2available_ensembles[language] = []
    for language, row in available_ensembles:
        lang2available_ensembles[language].append(row)
    predictions_made = set()
    run_index = 0
    while True:
        random.shuffle(languages)
        parallel_runs = []
        for language in languages:
            available_ensembles = lang2available_ensembles[language]
            if run_index >= len(available_ensembles):
                continue
            run = available_ensembles[run_index]
            parallel_runs.append(enumerate_rows(run, language))
        if not parallel_runs:
            break
        for rows in itertools.zip_longest(parallel_runs):
            for row in rows:
                if row is not None:
                    retval.append(row)
        run_index += 1
    return retval

def print_budget_details_header(languages):
    global l2text
    header = 'index language duration score budget days'.split()
    for language in languages:
        header.append(l2text[language])
    if len(languages) > 1:
        header.append('Average LAS')
    print('\t'.join(header))

def stable_sum(numbers):
    # add numbers in order of size to reduce numeric errors and to
    # stabalise the total for the same set presented in different orders
    if not numbers:
        return 0.0
    numbers.sort()
    assert numbers[0] >= 0.0   # implementation below not suitable for negative numbers
    total = 0.0
    for duration in numbers:
        total += duration
    return total

def get_average_best_score(best_score, languages):
    if len(languages) == 1:
        best_score = best_score[languages[0]]
    else:
        total = 0.0
        for language in best_score:
            total += best_score[language]
        best_score = total / float(len(best_score))
    return best_score

def print_budget_details_row(languages, index, language, duration, score, best_score, budget):
    row = []
    row.append('%d\t%s\t%15.1f\t%7.2f' %(index, language, duration, score))
    budget = stable_sum(budget)
    days = budget / (24*3600.0)
    row.append('%15.1f\t%15.5f' %(budget, days))
    for column in languages:
        if column == language:
            row.append('%7.2f' %score)
        else:
            row.append('')
    best_score = get_average_best_score(best_score, languages)
    row.append('%7.2f' %best_score)
    print('\t'.join(row))

def get_budget_and_best_score(selection, languages, cache = None, print_details = False):
    start = 0
    cached_budget = 0.0
    if cache is not None:
        n_rows = len(selection)
        if n_rows in cache:
            return cache[n_rows][:2]
        for key in cache:
            if key <= n_rows and key > start:
                start = key
        if start and not print_details:
            cached_budget, cached_score, cached_scores = cache[start]
    budget = []
    lang2best_score = {}
    for language in languages:
        lang2best_score[language] = 0.0
    if print_details:
        print_budget_details_header(languages)
    index = 0
    for language, duration, score in selection[start:]:
        budget.append(duration)
        if score > lang2best_score[language]:
            lang2best_score[language] = score
        if print_details:
            print_budget_details_row(
                languages, index, language, duration, score,
                lang2best_score, budget
            )
        index += 1
    if start > 0:
        for language in cached_scores:
            score = cached_scores[language]
            if score > lang2best_score[language]:
                lang2best_score[language] = score
    best_score = get_average_best_score(lang2best_score, languages)
    budget = stable_sum(budget)
    # convert seconds to days
    budget /= (24*3600)
    budget += cached_budget
    if cache is not None:
        cache[n_rows] = (budget, best_score, lang2best_score)
    return (budget, best_score)

def print_run(run, prefix = ''):
     for t, iteration in enumerate(run):
         print('%st=%d' %(prefix, t))
         for table in iteration:
             print('%s\t%r' %(prefix, table))

# get total budget and show an example schedule (if requested with --show-schedule)

if opt_show_schedule:
    print('Runs:')
    for r_index, run in available_ensembles:
        print('[%d]' %r_index)
        print_run(run, '\t')
    print('Example schedule for the full budget:')
random.shuffle(available_ensembles)  # for a realistic example
available_models = expand_steps(available_ensembles, languages)
total_budget, highest_score = get_budget_and_best_score(
    available_models, languages, print_details = opt_show_schedule
)

assert total_budget > 0

smallest_budget = total_budget
lowest_score = highest_score
start_index = 0
for row in available_ensembles:
    selection = expand_steps([row], languages)
    budget, score = get_budget_and_best_score(selection, languages)
    if not score or not budget:
         print('Run with zero budget or score:')
         print('budget', budget)
         print('score', score)
         print('run:', row)
         print_run(row)
         print('expanded to:')
         for i, row in enumerate(selection):
             print('[%d]\t%r' %(i, row))
         continue
    if budget < smallest_budget:
        smallest_budget = budget
    if score < lowest_score:
        lowest_score = score

budget_ratio = total_budget/smallest_budget
score_ratio = highest_score/lowest_score
print('budget range:', smallest_budget, 'to', total_budget, '(ratio %.2f)' %budget_ratio)
print('score range:', lowest_score, 'to', highest_score, '(ratio %.2f)' %score_ratio)

# prepare the budget points for which we output score distributions

print('Number of bins:', number_of_bins)

bin_budget_ratio = budget_ratio/1.001   # set lowest ceiling slightly above lowest budget
bin_ceiling_multiplier = bin_budget_ratio**(1.0/(number_of_bins-1))
bin_ceilings = []
next_bin_ceiling = total_budget
for _ in range(number_of_bins):
    bin_ceilings.append(next_bin_ceiling)
    next_bin_ceiling /= bin_ceiling_multiplier

bin_ceilings.sort()
print('Bins:')
header = 'index floor ceiling'.split()
print('\t'.join(header))
last_ceiling = 0.0
for index in range(number_of_bins):
    ceiling = bin_ceilings[index]
    print('%d\t%15.5f\t%15.5f' %(index, last_ceiling, ceiling))
    last_ceiling = ceiling

# collect samples

print('Number of samples:', number_of_samples)

# initialise stats
ceiling2scores = {}
for ceiling in bin_ceilings:
    ceiling2scores[ceiling] = []

last_run_with_progress = 0
current_run = 0
last_state = None
last_verbose = 0.0
def satisfied_with_samples(ceiling2scores, target_n):
    global last_run_with_progress
    global current_run
    global last_state
    global bin_ceilings
    global last_verbose
    current_run += 1
    current_state = []
    highest_non_ready = -1
    for ceiling in bin_ceilings:
        ready = len(ceiling2scores[ceiling])
        current_state.append(ready)
        if ready < target_n and ready > highest_non_ready:
            highest_non_ready = ready
    current_state = tuple(current_state)
    now = time.time()
    if now > last_verbose + 1.0:
        print('Run %d: %d of %d bins ready, highest non-ready bin has %.1f%% of samples' %(
            current_run, current_state.count(target_n), len(current_state),
            100.0 * highest_non_ready / float(target_n))
        )
        sys.stdout.flush()
        last_verbose = now
    if current_state != last_state:
        last_run_with_progress = current_run
        last_state = current_state
    if min(current_state) == target_n:
        return True
    if current_run - last_run_with_progress > 500:
        # not making progress, give up
        return True
    return False

n_models = len(available_models)

# main loop
while not satisfied_with_samples(ceiling2scores, number_of_samples):
    random.shuffle(available_ensembles)
    selection = expand_steps(available_ensembles, languages)   # clone and expand
    cache = {}
    assert len(selection) == n_models
    for ceiling in reversed(bin_ceilings):
        if len(ceiling2scores[ceiling]) >= number_of_samples:
            # collected enough samples for this ceiling
            continue
        # check that ceiling is feasible with current shuffle
        min_budget, _ = get_budget_and_best_score([selection[0]], languages)
        if min_budget > ceiling:
            # not a single model finishes within the budget
            # --> score is 0
            ceiling2scores[ceiling].append(0.0)
            continue
        n_selected = len(selection)
        if get_budget_and_best_score(selection, languages, cache)[0] > ceiling:
            # budget of current selection is too high
            # --> perform a binary search for the highest number of
            #     models that can be selected such that budget fits
            #     into ceiling
            lower = 1
            while n_selected - lower > 1:
                m_selected = (lower+n_selected)//2
                candidate = selection[:m_selected]
                budget, score = get_budget_and_best_score(candidate, languages, cache)
                if budget <= ceiling:
                    lower = m_selected
                else:
                    n_selected = m_selected
            n_selected = lower
            selection = selection[:n_selected]
        assert n_selected > 0
        budget, score = get_budget_and_best_score(selection, languages, cache)
        assert budget <= ceiling
        ceiling2scores[ceiling].append(score)

print('Scores in each bin:')
header = 'index ceiling n min avg max'.split()
print('\t'.join(header))
for index in range(number_of_bins):
    ceiling = bin_ceilings[index]
    scores = ceiling2scores[ceiling]
    if scores:
        n = len(scores)
        min_score = min(scores)
        avg_score = sum(scores)/float(n)
        max_score = max(scores)
        print('%d\t%15.5f\t%d\t%7.4f\t%7.4f\t%7.4f' %(index, ceiling, n, min_score, avg_score, max_score))
    else:
        print('%d\t%15.5f\t0' %(index, ceiling))

print('Duration:', time.time()-start_t)
