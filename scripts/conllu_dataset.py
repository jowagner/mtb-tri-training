#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import collections
import os
import subprocess
import sys

import basic_dataset

id_column = 0
token_column = 1
pos_column = 3
head_column = 6
label_column = 7


class ConlluSentence(basic_dataset.Sentence):

    def __init__(self):
        basic_dataset.Sentence.__init__(self)
        self.rows = []
        self.token2row = []

    def __getitem__(self, index):
        return self.rows[self.token2row[index]]

    def collect_labels(self, labelset, column):
        for row in self:
            label = row[column]
            if label != '@@MISSING@@':
                labelset[label] = None

    def __len__(self):
        return len(self.token2row)

    def __repr__(self):
        return '<ConlluSentence %r %r>' %(
            self.token2row,
            self.rows,
        )

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
        for row in self.rows:
            copy.append('\t'.join(row))
        return copy

    def is_missing(self, index, column):
        return self[index][column] == '@@MISSING@@'

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
        self[index][column] = '@@MISSING@@'

    def get_vector_representation(self):
        # TODO: return 4096 dimensional vector
        # counting each n-gram hash value
        raise NotImplementedError

class ConlluDataset(basic_dataset.Dataset):

    def __init__(self):
        basic_dataset.Dataset.__init__(self)

    def __repr__(self):
        return '<Conllu with %d sentences>' %(
            len(self),
        )

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
            if '@@MISSING@@' in row:
                for column, cell in enumerate(row):
                    if column:
                        f_out.write('\t')
                    if column < 2:
                        f_out.write(cell)
                    elif cell == '@@MISSING@@':
                        f_out.write('_')
                    else:
                        f_out.write(cell)
            else:
                f_out.write('\t'.join(row))
            f_out.write('\n')
        f_out.write('\n')

def get_tbname(tbid, treebank_dir, tbmapfile = None):
    if not tbmapfile:
        candidate_file = '%s/tbnames.tsv' %treebank_dir
        if os.path.exists(candidate_file):
            tbmapfile = candidate_file
        elif 'UD_TBNAMES' in os.environ:
            tbmapfile = os.environ['UD_TBNAMES']
    if tbmapfile:
        f = open(tbmapfile, 'r')
        while True:
            line = f.readline()
            if not line:
                break
            fields = line.split()
            if tbid == fields[1]:
                f.close()
                return fields[0]
        f.close()
        raise ValueError('TBID %r not found in %r' %(tbid, tbmapfile))
    if treebank_dir:
        # scan treebank folder
        lcode = tbid.split('_')[1]
        for entry in os.listdir(treebank_dir):
            # is this a good candidate entry?
            if entry.startswith('UD_') and lcode in entry.lower():
                # look inside
                if os.path.exists('%s/%s/%s-ud-test.conllu' %(
                    treebank_dir, entry, tbid
                )):
                    return entry
        raise ValueError('TBID %r not found in %r (must have test set)' %(tbid, treebank_dir))
    raise ValueError('TBID %r not found (need map file or treebank dir)' %tbid)

def load(dataset_id,
    treebank_dir = None,
    tbname = None,
    tbmapfile = None,
    load_tr = True, load_dev = True, load_test = True,
    mode = 'load',
    only_get_path = False
):
    tbid = dataset_id
    tr, dev, test = None, None, None
    if not treebank_dir:
        treebank_dir = os.environ['UD_TREEBANK_DIR']
    if not tbname:
        tbname = get_tbname(tbid, treebank_dir, tbmapfile)
    if load_tr:
        filename = '%s/%s/%s-ud-train.conllu' %(treebank_dir, tbname, tbid)
        if only_get_path:
            tr = filename
        elif os.path.exists(filename):
            tr = basic_dataset.load_or_map_from_filename(
                ConlluDataset(), filename, mode
            )
        else:
            print('Warning: %r not found' %filename)
    if load_dev:
        filename = '%s/%s/%s-ud-dev.conllu' %(treebank_dir, tbname, tbid)
        if only_get_path:
            dev = filename
        elif os.path.exists(filename):
            dev = basic_dataset.load_or_map_from_filename(
                ConlluDataset(), filename, mode
            )
        else:
            print('Warning: %r not found' %filename)
    if load_test:
        filename = '%s/%s/%s-ud-test.conllu' %(treebank_dir, tbname, tbid)
        if only_get_path:
            test = filename
        elif os.path.exists(filename):
            test = basic_dataset.load_or_map_from_filename(
                ConlluDataset(), filename, mode
            )
        else:
            print('Warning: %r not found' %filename)
    return tr, dev, test

def new_empty_set():
    return ConlluDataset()

def get_target_columns():
    return [2,3,4,5,6,7,8]

def get_filename_extension():
    ''' recommended extension for output files '''
    return '.conllu'

def combine(prediction_paths, output_path, combiner_dir = None, seed = '42'):
    ''' combine (ensemble) the given predictions
        into a single prediction
    '''
    if not combiner_dir:
        combiner_dir = os.environ['CONLLU_COMBINER_DIR']
    command = []
    command.append('%s/parser.py' %combiner_dir)
    command.append('--outfile')
    command.append(output_path)
    command.append('--overwrite')
    command.append('--prune-labels')
    command.append('--seed')
    command.append(seed)
    for prediction_path in prediction_paths:
        command.append(prediction_path)
    print('Running', command)
    sys.stderr.flush()
    sys.stdout.flush()
    subprocess.call(command)

def main():
    import random
    import sys
    if len(sys.argv) < 2 or sys.argv[1][:3] in ('-h', '--h'):
        print('usage: $0 $NUMBER_SENTENCES {load|map} {dropoutsample|shuffle} < in.conllu > out.conllu')
        sys.exit(1)
    max_sentences = int(sys.argv[1])
    mode = sys.argv[2]
    dataset = ConlluDataset()
    dataset.load_or_map_file(sys.stdin, max_sentences, mode)
    if sys.argv[3] == 'shuffle':
        dataset.shuffle(random)
        dataset.save_to_file(sys.stdout)
        return
    dropout = basic_dataset.SentenceDropout(random,
            [pos_column, head_column, label_column],
            [0.2,        0.8,         0.5]
    )
    sample = basic_dataset.Sample(dataset, random, sentence_modifier = dropout)
    sample.save_to_file(sys.stdout)

if __name__ == "__main__":
    main()

