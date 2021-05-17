#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

from __future__ import print_function

import hashlib
import random
import sys
import time

from distribution import Distribution

start_t = time.time()

if sys.argv[1] == '--show-schedule':
    opt_show_schedule = True
    del sys.argv[1]
else:
    opt_show_schedule = False

language = sys.argv[1]
number_of_bins = int(sys.argv[2])
number_of_samples = int(sys.argv[3])
random.seed(int(hashlib.sha256(sys.argv[4]).hexdigest(), 16))

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
    'p': 'bootstrap samples of labelled data (300% of size of data)',
    't': '300% of labelled data, i.e. concatenation of 3 copies',
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

available_ensembles = []

parsers_and_languages = []
for parser in parsers:
    for language in languages:
        parsers_and_languages.append((parser, language))

for parser, language in parsers_and_languages:
    for sample in s2text.keys():
        distribution = Distribution(
            language, parser, sample, with_info = True, quiet = True
        )
        for row in distribution.info:
            if len(row) != 17:
                raise ValueError('row %r' %row)
            available_ensembles.append((language, row))

def add_ensemble_to_budget(budget, row, language):
    duration = 21*0.3      # estimate for combiner and eval runs (10x parallel)
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

def expand_steps(available_ensembles, languages):
    retval = []
    # re-organise languages in separate streams
    lang2available_ensembles = {}
    for language in languages:
        lang2available_ensembles[language] = []
    for language, row in available_ensembles:
        lang2available_ensembles[language].append(row)
    predictions_made = set()
    row_index = 0
    learner_indices = [14, 15, 16]
    completed_ensembles = set()
    candidate_rows = get_candidate_rows(lang2available_ensembles)
    while True:
        found_data_at_index = False
        # [0]=ensemble LAS, [1]=combination code,
        # [2,3,4]=individual learner LAS,
        # [5]=ensemble stddev, [6,7]=ensemble min and max,
        # [8,9,10]=seeds, [11,12,13]=duration model training,
        # [14,15,16]=prediction files (connlu or connlu.bz2)
        random.shuffle(learner_indices)
        # add individual learners at this row index
        for index in learner_indices:
            random.shuffle(languages)
            for language in languages:
                available_ensembles = lang2available_ensembles[language]
                if row_index >= len(available_ensembles):
                    continue
                found_data_at_index = True
                row = available_ensembles[row_index]
                prediction = row[index]
                if prediction not in predictions_made:
                    duration = float(row[index-3])  # model training time
                    duration += 30.0                # estimate for prediction time
                    score    = float(row[index-12]) # individual score
                    retval.append((language, duration, score))
                    predictions_made.add(prediction)
                    bring_eligble_ensembles_forward(
                        row_index+1, language, completed_ensembles,
                        available_ensembles, retval, predictions_made,
                        candidate_rows, prediction
                    )
        if not found_data_at_index:
            break
        # add ensemble(s) at this row index
        random.shuffle(languages)
        for language in languages:
            if (language, row_index) in completed_ensembles:
                # this ensemble has been completed earlier
                continue
            available_ensembles = lang2available_ensembles[language]
            if row_index >= len(available_ensembles):
                continue
            row = available_ensembles[row_index]
            add_ensemble_to_budget(retval, row, language)
            # no need to record this ensemble in `completed_ensembles`
            # as we never consider this key again
        row_index += 1
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

# get total budget and show an example schedule (if requested with --show-schedule)

if opt_show_schedule:
    print('Example schedule for the full budget:')
random.shuffle(available_ensembles)  # for a realistic example
available_models = expand_steps(available_ensembles, languages)
total_budget, highest_score = get_budget_and_best_score(
    available_models, languages, print_details = opt_show_schedule
)

smallest_budget = total_budget
lowest_score = highest_score
start_index = 0
for row in available_ensembles:
    selection = expand_steps([row], languages)
    budget, score = get_budget_and_best_score(selection, languages)
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
