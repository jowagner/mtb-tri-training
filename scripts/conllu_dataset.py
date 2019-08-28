#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import collections

import basic_dataset

id_column = 0
token_column = 1
pos_column = 3
head_column = 6


class ConlluSentence(basic_dataset.Sentence):

    def __init__(self):
        basic_dataset.Sentence.__init__(self)
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
        # remove linebreak from last cell
        fields[-1] = fields[-1].rstrip()
        #
        r_index = len(self.rows)
        self.rows.append(fields)
        # check whether this is a UD token
        token_id = fields[id_column]
        if token_id.startswith('#') \
        or '-' in token_id \
        or '.' in token_id:
            return
        # record UD token
        self.token2row.append(r_index)

    def clone(self):
        copy = ConlluSentence()
        for row in self.rows():
            copy.append('\t'.join(row))
        return copy

    def is_missing(self, index, column):
        return self[index][column] == '_'

    def set_label(self, index, column, new_label):
        self[index][column] = new_label

    def possible_labels(self, index, column):
        if column != head_column:
            # See SentenceCompleter for the normal way
            # to supply possible labels.
            raise ValueError('Sentence does not know labels for column %d.' %column)
        retval = list(range(len(self)+1))
        del retval[index+1]
        return map(lambda x: '%d' %x, retval)

    def unset_label(self, index, column):
        self[index][column] = '_'


class ConlluDataset(basic_dataset.Dataset):

    def __init__(self):
        basic_dataset.Dataset.__init__(self)

    def read_sentence(self, f_in):
        sentence = None
        while True:
            line = f_in.readline()
            if not line:
                if sentence is not None:
                    raise ValueError('unexpected end of file in conllu sentence')
                break
            elif sentence is None:
                sentence = ConlluSentence()
            if line.isspace():
                break
            sentence.append(line)
        return sentence

    def write_sentence(self, f_out, sentence):
        for row in sentence.rows:
            # incomplete sentences should be completed by the caller,
            # e.g. save_to_file(), see basic_dataset.SentenceCompleter
            f_out.write('\t'.join(row))
            f_out.write('\n')
        f_out.write('\n')


def main():
    import sys
    max_sentences = int(sys.argv[1])
    mode = sys.argv[2]
    dataset = ConlluDataset()
    dataset.load_or_map_file(sys.stdin, max_sentences, mode)
    dataset.save_to_file(sys.stdout)

if __name__ == "__main__":
    main()

