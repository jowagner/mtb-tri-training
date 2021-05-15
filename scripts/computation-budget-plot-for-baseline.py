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

from collections import defaultdict
import random
import sys

from distribution import Distribution

language = sys.argv[1]
number_of_bins = int(sys.argv[2])

l2text = {
    'e': 'English',
    'h': 'Hungarian',
    't': 'English-LinEs',
    'u': 'Uyghur',
    'v': 'Vietnamese',
}
p2text = {
    'f': 'udpipe-future',
    'g': '+fasttext',
    'h': '+elmo',
    'i': 'udpf+fasttext+mBERT',
}
v2text = {
    '-': 'no, using copy of seed sample',
    'o': 'yes, labelled data oversampled to size of unlabelled data',
}
s2text = {
    '-': 'bootstrap samples of labelled data (100% of size of data)',
    'p': 'bootstrap samples of labelled data (300% of size of data)',
    't': '300% of labelled data, i.e. concatenation of 3 copies',
    'w': 'permutations of labelled data, i.e. shuffling the data',
    'x': '250% of labelled data, i.e. concatenation of 2 1/2 copies',
}
d2text = {
    '-': 'use all (current and all previous data)',
    'v': 'vanilla approach (only using current iteration\'s data)',
    'y': 'last 5 iterations',
    'o': 'decaying with factor 0.71',
    'z': 'decaying with factor 0.50',
    'a': 'use all, using all labelled data',
    'u': 'vanilla, using all labelled data',
    'r': 'decaying with factor 0.50, using all labelled data',
}

parsers = list(p2text.keys())

if language != 'e':
    parsers = 'fgh'  # at the moment, mBERT only ready for English

available_ensembles = []

for parser in parsers:
    for sample in s2text.keys():
        distribution = Distribution(
            language, parser, sample, with_info = True, quiet = True
        )
        for row in distribution.info:
            if len(row) != 17:
                raise ValueError('row %r' %row)
            available_ensembles.append(row)

def get_budget_and_best_score(selection):
    predictions_made = set()
    budget = []
    best_score = 0.0
    for row in selection:
        # [0]=ensemble LAS, [1]=combination code,
        # [2,3,4]=individual learner LAS,
        # [5]=ensemble stddev, [6,7]=ensemble min and max,
        # [8,9,10]=seeds, [11,12,13]=duration model training,
        # [14,15,16]=prediction files (connlu or connlu.bz2)
        for index in (14,15,16):
            prediction = row[index]
            if prediction not in predictions_made:
                duration = float(row[index-3])
                budget.append(duration)  # model training time
                budget.append(30.0)      # estimate for prediction time
                predictions_made.add(prediction)
        budget.append(0.3)  # estimate for combiner and eval runs (10x parallel)
        score = float(row[0])
        if score > best_score:
            best_score = score
    # add numbers in order of size to reduce numeric errors and to
    # stabalise the total for the same set presented in different orders
    budget.sort()
    total = 0.0
    for duration in budget:
        total += duration
    return (total, best_score)

total_budget, highest_score = get_budget_and_best_score(available_ensembles)

smallest_budget = total_budget
lowest_score = highest_score
for row in available_ensembles:
    budget, score = get_budget_and_best_score([row])
    if budget < smallest_budget:
        smallest_budget = budget
    if score < lowest_score:
        lowest_score = score

budget_ratio = total_budget/smallest_budget
score_ratio = highest_score/lowest_score
print('budget range:', smallest_budget, 'to', total_budget, '(ratio %.2f)' %budget_ratio)
print('score range:', lowest_score, 'to', highest_score, '(ratio %.2f)' %score_ratio)

# prepare the budget points for which we output score distributions

bin_budget_ratio = total_budget/(smallest_budget+12.3)
bin_ceiling_multiplier = bin_budget_ratio**(1.0/(number_of_bins-1))
bin_ceilings = []
next_bin_ceiling = total_budget
for _ in range(number_of_bins):
    bin_ceilings.append(next_bin_ceiling)
    next_bin_ceiling /= bin_ceiling_multiplier

bin_ceilings.sort()
print('Bins:')
last_ceiling = 0.0
for index in range(number_of_bins):
    ceiling = bin_ceilings[index]
    print('[%d] %.3f to %.3f' %(index, last_ceiling, ceiling))
    last_ceiling = ceiling
