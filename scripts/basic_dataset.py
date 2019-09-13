#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import collections
import hashlib
import random
import StringIO

class Sentence(collections.Sequence):

    def __init__(self):
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def collect_labels(self, labelset, column):
        ''' For each label in the given column, add the
            label as a key to the dictionary `labelset`
        '''
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def append(self, line):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    def is_missing(self, index, column):
        return False

    def set_label(self, index, column, new_label):
        raise NotImplementedError

    def possible_labels(self, index, column):
        raise NotImplementedError

    def unset_label(self, index, column):
        raise NotImplementedError

    def get_vector_representation(self):
        raise NotImplementedError


class Dataset(collections.Sequence):

    """ Abstract base class for data sets.
    """

    def __init__(self):
        self.sentences = []    # see __getitem__() below for format
        self.files = []

    def __getitem__(self, index):
        f_index, info = self.sentences[index]
        if f_index < 0:
            return info
        else:
            f_in = self.files[f_index]
            f_in.seek(info)
            return self.read_sentence(f_in)

    def collect_labels(self, labelset, column):
        for sentence in self:
            sentence.collect_labels(labelset, column)

    def __len__(self):
        return len(self.sentences)

    def clone(self):
        retval = Dataset()
        # make new lists so that shuffling the clone
        # does not affect self
        retval.sentences = self.sentences[:]
        retval.files = self.files[:]
        return retval

    def append(self, sentence):
        self.sentences.append((-1, sentence))

    def shuffle(self, rng):
        rng.shuffle(self.sentences)

    def load_file(self, f_in, max_sentences = None):
        """ Append up to `max_sentences` sentences from file `f_in`
            to the data set. No limit if `max_sentences` is `None`.
        """
        self.load_or_map_file(f_in, max_sentences, 'load')

    def map_file(self, f_in, max_sentences = None):
        """ Append up to `max_sentences` sentences from file `f_in`
            to the data set. No limit if `max_sentences` is `None`.
            The file is scanned for start positions of sentences
            and only references are stored in memory. Data will be
            re-read each time it is needed. The file must not be
            modified or closed while it is mapped.
        """
        self.load_or_map_file(f_in, max_sentences, 'map')

    def load_or_map_file(self, f_in, max_sentences, mode):
        if mode == 'map':
            f_index = len(self.files)
            self.files.append(f_in)
        else:
            f_index = -1
        added = 0
        while True:
            if max_sentences is not None and added == max_sentences:
                break
            if mode == 'map':
                info = f_in.tell()
            sentence = self.read_sentence(f_in)
            if not sentence:
                break
            if mode == 'load':
                info = sentence
            self.sentences.append((f_index, info))
            added += 1
        return added

    def save_to_file(self, f_out,
        sentence_filter    = None,
        sentence_completer = None
    ):
        for sentence in self:
            if sentence_filter is not None \
            and sentence_filter(sentence):
                # skip this sentence
                continue
            if sentence_completer is not None:
                sentence = sentence_completer(sentence)
            self.write_sentence(f_out, sentence)

    def hexdigest(self):
        h = hashlib.sha512()
        for sentence in self:
            f = StringIO.StringIO()
            self.write_sentence(f, sentence)
            f.seek(0)
            h.update(f.read())
        return h.hexdigest()

    def read_sentence(self, f_in):
        raise NotImplementedError

    def write_sentence(self, f_out, sentence):
        raise NotImplementedError

    def get_number_of_items(self):
        count = 0
        for sentence in self:
            count += len(sentence)
        return count


class SentenceCompleter:

    def __init__(self, rng, target_columns, target_labelsets):
        self.rng = rng
        self.target_columns = target_columns
        self.target_labelsets = target_labelsets

    def pick_label(self, sentence, item_index, tc_index, column):
        labelset = self.target_labelsets[tc_index]
        if not labelset:
            labelset = sentence.possible_labels(item_index, column)
        return self.rng.choice(labelset)

    def __call__(self, sentence):
        retval = None
        for item_index in range(len(sentence)):
            for tc_index, column in enumerate(self.target_columns):
                if sentence.is_missing(item_index, column):
                    if not retval:
                        retval = sentence.clone()
                    new_label = self.pick_label(
                        sentence, item_index, tc_index, column
                    )
                    retval.set_label(item_index, column, new_label)
        if not retval:
            return sentence
        return retval


class SentenceDropout:

    def __init__(self, rng, target_columns, dropout_probabilities):
        self.rng = rng
        self.target_columns = target_columns
        self.dropout_probabilities = dropout_probabilities

    def __call__(self, sentence):
        retval = sentence.clone()
        for item_index in range(len(sentence)):
            for tc_index, column in enumerate(self.target_columns):
                dropout_probability = self.dropout_probabilities[tc_index]
                if self.rng.random() < dropout_probability:
                    retval.unset_label(item_index, column)
        return retval


class SentenceFilter:

    def __init__(self, target_columns,
        min_labelled = None, max_unlabelled = None,
        min_percentage_labelled = None,
        max_percentage_unlabelled = None
    ):
        self.target_columns = target_columns
        self.min_labelled   = min_labelled
        self.max_unlabelled = max_unlabelled
        self.min_percentage = min_percentage_labelled
        self.max_percentage = max_percentage_unlabelled

    def __call__(self, sentence):
        ''' returns True if the sentence should be skipped '''
        num_items = len(sentence)
        for tc_index, column in enumerate(self.target_columns):
            num_labelled = 0
            for item_index in range(num_items):
                if not sentence.is_missing(item_index, column):
                    num_labelled += 1
            if self.min_labelled \
            and num_labelled < self.min_labelled[tc_index]:
                return True
            num_unlabelled = num_items - num_labelled
            if self.max_unlabelled \
            and num_unlabelled > self.max_unlabelled[tc_index]:
                return True
            percentage = 100.0 * num_labelled / float(num_items)
            if self.min_percentage \
            and percentage < self.min_percentage[tc_index]:
                return True
            percentage = 100.0 * num_unlabelled / float(num_items)
            if self.max_percentage \
            and percentage > self.max_percentage[tc_index]:
                return True
        return False


class Concat(Dataset):

    def __init__(self, datasets, sentence_modifier = None):
        self.datasets = datasets
        self.sentence_modifier = sentence_modifier
        self.sentences = []
        for ds_index, dataset in enumerate(datasets):
            if dataset is None:
                continue
            for d_index in range(len(dataset)):
                self.sentences.append((ds_index, d_index))

    def __getitem__(self, index):
        ds_index, d_index = self.sentences[index]
        sentence = self.datasets[ds_index][d_index]
        if self.sentence_modifier is not None:
            sentence = self.sentence_modifier(sentence)
        return sentence

    def clone(self):
        retval = Concat([])
        # make new lists so that shuffling the clone
        # does not affect self
        retval.sentences = self.sentences[:]
        retval.datasets = self.datasets[:]
        retval.sentence_modifier = self.sentence_modifier
        return retval

    def append(self, item):
        raise ValueError('Cannot append to concatenation')

    def load_or_map_file(self, *args):
        raise ValueError('Cannot load data into concatenation')

    def write_sentence(self, f_out, sentence):
        self.datasets[0].write_sentence(f_out, sentence)


class Sample(Dataset):

    def __init__(self, dataset, rng, size = None, percentage = None,
        with_replacement = True,
        sentence_modifier = None,
        diversify_attempts = 1,
        disprefer = {}
    ):
        if size and percentage:
            raise ValueError('Must not specify both size and percentage.')
        if percentage:
            size = int(0.5+percentage*len(dataset)/100.0)
        self.dataset = dataset
        self.is_vectorised = False
        self.sentence_modifier = sentence_modifier
        self.with_replacement  = with_replacement
        self.reset_sample(rng, size, diversify_attempts, disprefer)


    def _get_preferred_d_indices(self, d_size, size, disprefer):
        if size >= d_size or not disprefer:
            # use all data
            return list(range(d_size))
        # stratify data according to
        # how strongly items are dispreferred
        level2indices = {}
        max_level = 0
        for d_index in range(d_size):
            try:
                level = disprefer[d_index]
            except KeyError:
                level = 0
            if level not in level2indices:
                level2indices[level] = []
            level2indices[level].append(d_index)
            if level > max_level:
                max_level = level
        # select as much data as needed
        # starting with the lowest levels
        retval = []
        level = 0
        while len(retval) < size:
            assert level <= max_level, 'Missing some data after stratification.'
            try:
                indices = level2indices[level]
            except KeyError:
                indices = []
            retval += indices
            level += 1
        return retval

    def reset_sample(
        self, rng, size = None,
        diversify_attempts = 1,
        disprefer = {}
    ):
        if self.with_replacement and disprefer:
            # not clear how this should be implemented,
            # e.g. with what probability dispreferred
            # items should be picked
            raise NotImplementedError
        d_size = len(self.dataset)
        if size is None:
            size = d_size
        if not self.with_replacement:
            permutation = self._get_preferred_d_indices(
                d_size, size, disprefer
            )
            p_size = len(permutation)
            rng.shuffle(permutation)
        self.sentences = []
        remaining = size
        while remaining:
            candidates = []
            for attempt in range(diversify_attempts):
                if self.with_replacement:
                    d_index = rng.randrange(d_size)
                else:
                    d_index = permutation[(size-remaining) % p_size]
                if diversify_attempts == 1 or not self.sentences:
                    # no choice
                    priority = 0
                else:
                    priority = -self._nearest_neighbour_distance(d_index)
                candidates.append((priority, attempt, d_index))
            candidates.sort()
            d_index = candidates[0][-1]
            self.sentences.append(d_index)
            remaining -= 1

    def _nearest_neighbour_distance(self, d_index):
        if not self.is_vectorised:
            self._vectorise()
        nn_distance = self._vector_distance(self.sentence[0], d_index)
        for candidate_index in self.sentences[1:]:
            distance = self._vector_distance(candidate_index, d_index)
            if distance < nn_distance:
                nn_distance = distance
        return nn_distance

    def _vectorise(self):
        self.vectors = []
        for d_index, sentence in enumerate(self.dataset):
            self.vectors.append(sentence.get_vector_representation())
        self.is_vectorised = True

    def __getitem__(self, index):
        d_index = self.sentences[index]
        sentence = self.dataset[d_index]
        if self.sentence_modifier is not None:
            sentence = self.sentence_modifier(sentence)
        return sentence

    def indices(self):
        return self.sentences

    def clone(self):
        retval = Sample([], random)
        # make new lists so that shuffling the clone
        # or changing the subset with reset_sample(),
        # set_counts() or set_remaining() does not
        # affect self
        retval.sentences = self.sentences[:]
        retval.dataset = self.dataset
        retval.is_vectorised = self.is_vectorised
        if self.is_vectorised:
            retval.vectors = self.vectors
        retval.sentence_modifier = self.sentence_modifier
        retval.with_replacement  = self.with_replacement
        return retval

    def append(self, item):
        raise ValueError('Cannot append to sample')

    def load_or_map_file(self, *args):
        raise ValueError('Cannot load data into sample')

    def get_counts(self):
        retval = len(self.dataset) * [0]
        for d_index in self.sentences:
            retval[d_index] += 1
        return retval

    def set_counts(self, rng, counts):
        self.sentences = []
        for d_index, count in enumerate(counts):
            for _ in count:
                self.sentences.append(d_index)
        self.shuffle(rng)

    def set_remaining(self, rng):
        '''
        Make this dataset the subset not selected by the current sample.
        '''
        counts = self.get_counts()
        self.sentences = []
        for d_index, count in enumerate(counts):
            if not count:
                self.sentences.append(d_index)
        self.shuffle(rng)

    def write_sentence(self, f_out, sentence):
        self.dataset.write_sentence(f_out, sentence)


def load_or_map_from_filename(data, filename, mode = 'load'):
    f_in = open(filename, 'r')
    data.load_or_map_file(f_in, None, mode)
    if mode == 'load':
        f_in.close()
    return data

def load(dataset_id,
    load_tr = True, load_dev = True, load_test = True,
    mode = 'load'
):
    raise NotImplementedError


def new_empty_set():
    raise NotImplementedError

def get_target_columns():
    raise NotImplementedError

def get_filename_extension():
    ''' recommended extension for output files '''
    return '.data'

