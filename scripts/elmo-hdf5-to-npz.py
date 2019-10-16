#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# (C) 2018, 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# adapted from
# https://github.com/tira-io/ADAPT-DCU/blob/master/scripts/multi-en_ewt-subsets-3/get_elmo_sentence_features.py
# by same authors

# levenshteinDistance() copied from stackoverflow, see comment below

import os
import hashlib
import h5py
import numpy
import sys
import time


"""
Usage: ./elmo-hdf5-to-npz.py [--elmoformanylang input.conllu] input-tok-rep.hdf5 output-tok-rep.hdf5
"""

opt_elmoformanylanguages = None
if sys.argv[1][:17] == '--elmoformanylang':
    opt_elmoformanylanguages = sys.argv[2]
    del sys.argv[2]
    del sys.argv[1]

token_rep = h5py.File(sys.argv[1], 'r')
output_filename = sys.argv[2]

# The following function is from
# Salvador Dali's answer on
# https://stackoverflow.com/questions/2460177/edit-distance-in-python

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

# end of code from stackoverflow

keys = []
if opt_elmoformanylanguages:
    id_column    = 0
    token_column = 1
    sentences = []
    _file = open(opt_elmoformanylanguages, mode='r', encoding='utf-8')
    tokens = []
    while True:
        line = _file.readline()
        if not line:
            if tokens:
                raise ValueError('No end-of-sentence marker at end of file %r' %path)
            break
        if line.isspace():
            # apply same processing as in elmoformanylangs/__main__.py
            sent = '\t'.join(tokens)
            sent = sent.replace('.', '$period$')
            sent = sent.replace('/', '$backslash$')
            i = len(sentences)
            sentences.append((i, len(tokens), sent))
            tokens = []
        elif not line.startswith('#'):
            fields = line.split('\t')
            tok_id = fields[id_column]
            if not '.' in tok_id and not '-' in tok_id:
                tokens.append(fields[token_column])
    _file.close()
    num_sentences = len(sentences)
    exact_matches = 0
    wrong_length  = 0
    for i, num_tokens, sentence in sentences:
        if sentence in token_rep:
            exact_matches += 1
            best_key = sentence
            best_score = (100.0, 100.0, '')  # for debugging output below
        else:
            best_key = None
            best_score = (-1.0, -1.0, '')
            len1 = num_tokens
            for key in token_rep.keys():
                # most important that the sentence length is right
                len2 = len(key.split('\t'))
                score1 = 100.0/(1.0+abs(len1-len2))
                # content should also match as best as possible
                score2 = 100.0/(1.0+levenshteinDistance(key, sentence))
                info = '%d\n%s\n%s' %(i, key, sentence)
                tiebreaker = hashlib.sha256(info.encode('UTF-8')).hexdigest()
                if (score1, score2, tiebreaker) > best_score:
                    best_key   = key
                    best_score = (score1, score2, tiebreaker)
            if best_score[0] < 100.0:
                wrong_length += 1
        #print('[%d]' %i)
        #print('\tsentence =', sentence)
        #print('\tbest_key =', best_key)
        #print('\tscore    =', best_score)
        keys.append(best_key)
    if exact_matches < num_sentences:
        print('Elmo format conversion: only %d of %d matches were exact.' %(exact_matches, num_sentences))
    if wrong_length:
        print('Elmo format conversion: %d sentences have incorrect length' %wrong_length)

else:
    # hdf5 format of the `allennlp elmo` command
    num_sentences = len(token_rep) - 1  # -1 due to the extra key 'sentence_to_index'
    for i in range(num_sentences):
        keys.append('%d' %i)

npz_data = {}
for i, key in enumerate(keys):
    vectors = token_rep[key][()]
    npz_key = 'arr_%d' %i
    npz_data[npz_key] = vectors

token_rep.close()

numpy.savez(output_filename, **npz_data)

