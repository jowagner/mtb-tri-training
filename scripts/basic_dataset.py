#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import collections


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

    def save_to_file(self, f_out,
        sentence_filter    = None,
        sentence_completer = None
    ):
        for sentence in self:
            if sentence_filter is not None and sentence_filter(sentence):
                # skip this sentence
                continue
            if sentence_completer is not None:
                sentence = sentence_completer(sentence)
            self.write_sentence(f_out, sentence) 

    def read_sentence(self, f_in):
        raise NotImplementedError

    def write_sentence(self, f_out, sentence):
        raise NotImplementedError


class Sample(Dataset):

    def __init__(self, dataset, rng, size = None, percentage = None):
        if size and percentage:
            raise ValueError('Must not specify both size and percentage.')
        if percentage:
            size = int(0.5+percentage*len(dataset)/100.0)
        self.dataset = dataset
        self.reset_sample(rng, size)

    def reset_sample(self, rng, size = None):
        if not size:
            size = len(dataset)
        self.sentences = []
        remaining = size
        while remaining:
            d_index = rng.randrange(size)
            self.sentences.append(d_index)
            remaining -= 1

    def __getitem__(self, index):
        d_index = self.sentences[index]
        return self.dataset[d_index]

    def append(self, sentence):
        self.dataset.append(sentence)

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
