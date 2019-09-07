#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import collections
import hashlib
import StringIO

class Sentence(collections.Sequence):

    def __init__(self):
        pass

    def __getitem__(self, index):
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

    def __init__(self, datasets):
        self.datasets = datasets
        self.sentences = []
        for ds_index, dataset in enumerate(datasets):
            for d_index in range(len(dataset)):
                self.sentences.append((ds_index, d_index))

    def __getitem__(self, index):
        ds_index, d_index = self.sentences[index]
        return self.datasets[ds_index][d_index]

    def clone(self):
        retval = Concat([])
        # make new lists so that shuffling the clone
        # does not affect self
        retval.sentences = self.sentences[:]
        retval.datasets = self.datasets[:]
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
        sentence_modifier = None
    ):
        if size and percentage:
            raise ValueError('Must not specify both size and percentage.')
        if percentage:
            size = int(0.5+percentage*len(dataset)/100.0)
        self.dataset = dataset
        self.sentence_modifier = sentence_modifier
        self.with_replacement  = with_replacement
        self.reset_sample(rng, size)

    def reset_sample(self, rng, size = None):
        d_size = len(self.dataset)
        if size is None:
            size = d_size
        if not self.with_replacement:
            permutation = list(range(d_size))
            rng.shuffle(permutation)
        self.sentences = []
        remaining = size
        while remaining:
            if not self.with_replacement:
                if remaining >= d_size:
                    remaining -= d_size
                else:
                    permutation = permutation[:remaining]
                    remaining = 0
                self.sentences += permutation
                continue
            d_index = rng.randrange(d_size)
            self.sentences.append(d_index)
            remaining -= 1

    def __getitem__(self, index):
        d_index = self.sentences[index]
        sentence = self.dataset[d_index]
        if self.sentence_modifier is not None:
            sentence = self.sentence_modifier(sentence)
        return sentence

    def clone(self):
        retval = Sample([], random)
        # make new lists so that shuffling the clone
        # or changing the subset with reset_sample(),
        # set_counts() or set_remaining() does not
        # affect self
        retval.sentences = self.sentences[:]
        retval.dataset = self.dataset
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


def load(dataset_id,
    load_tr = True, load_dev = True, load_test = True,
    mode = 'load'
):
    raise NotImplementedError


def new_empty_set():
    raise NotImplementedError

