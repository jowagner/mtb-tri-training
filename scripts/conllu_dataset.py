#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

from basic_dataset import Dataset

class Sentence(collections.Sequence):

    def __init__(self):
        self.rows = []
        self.token2row = []

    def __getitem__(self, index):
        self.rows[self.token2row[index]]

    def __len__(self):
        return len(self.token2row)

    def append(self, line):
        fields = line.split('\t')
        if not fields:
            raise ValueError('cannot append empty line to conllu sentence')
        fields[-1] = fields[-1].rstrip()
        r_index = len(self.rows)
        self.rows.append(fields)
        # check whether this is a token
        if fields[0].startswith('#'):
            return
        # TODO: more checks
        raise NotImplementedError


class Conllu(Dataset):

    def __init__(self):
        Dataset.__init__(self)
        self.column2vocab = {}

    def read_sentence(self, f_in):
        raise NotImplementedError

    def write_sentence(self, f_out, sentence):
        raise NotImplementedError

