#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

from __future__ import print_function

import os
import subprocess
import sys

import common_udpipe_future

def request_npz(conllu_filename, npz_filename):
    task = ElmoNpzTask([conllu_filename, npz_filename], 'elmo-npz')
    task.submit()

class ElmoNpzTask(common_udpipe_future.Task):

    def start_processing(self):
        conllu_file, npz_name = self.command
        sentences = self.read_sentences(conllu_file)
        lcode = self.lcode
        for sentence in sentences:
            self.cache.request(sentence, npz_name, lcode)
        self.cache.submit()
        self.sentences = sentences
        self.number_of_sentences = len(sentences)

    def finished_processing(self):
        if self.sentences is None:
            # finished in a previous call
            return True
        _, npz_name = self.command
        if self.cache.npz2ready_count[npz_name] < self.number_of_sentences:
            # not ready yet
            return False
        # all vectors are ready
        sentences = self.sentences
        npz_data = {}
        for index, sentence in enumerate(sentences):
            vectors = self.cache.collect(sentence, npz_name)
            npz_key = 'arr_%d' %index
            npz_data[npz_key] = vectors
        # write npz file
        numpy.savez(npz_name + '.prep', **npz_data)
        os.rename(npz_name + '.prep', npz_name)
        # release data and mark as finished
        del self.cache.npz2ready_count[npz_name]
        self.sentences = None
        return True

    def read_sentences(self, conllu_file):
        id_column    = 0
        token_column = 1
        sentences = []
        try:
            # Python 3
            _file = open(conllu_file, mode='r', encoding='utf-8')
        except TypeError:
            # Python 2
            _file = open(conllu_file, mode='rb')
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
                sentences.append((i, len(tokens), tokens, sent))
                tokens = []
            elif not line.startswith('#'):
                fields = line.split('\t')
                tok_id = fields[id_column]
                if not '.' in tok_id and not '-' in tok_id:
                    tokens.append(fields[token_column])
        _file.close()
        return sentences

class ElmoCache:

    class Entry:
        def __init__(self, tokens, lcode):
            self.vectors = None
            self.last_access = 0.0
            self.access_time_synced = False
            self.vectors_on_disk    = False
            self.hdf5_in_progress   = False
            self.requesters         = []
            self.tokens             = tokens
            self.lcode              = lcode

    def __init__(self):
        self.npz2ready_count = {}
        self.key2entry = {}
        self.hdf5_tasks = []
        self.init_random()

    def request(self, sentence, npz_name, lcode):
        _, _, tokens, hdf5_key = sentence
        if not hdf5_key in self.key2npz_name:
            self.key2entry[hdf5_key] = ElmoCache.Entry(tokens, lcode)
        self.key2entry[hdf5_key].requesters.append(npz_name)

    def submit(self, npz_name):
        if npz_name in self.npz2ready_count:
            raise ValueError('Duplicate request to create %s' %npz_name)
        self.npz2ready_count[npz_name] = 0
        need_hdf5 = []
        number_of_entries = 0
        cache_hit_in_progress = 0
        cache_miss = 0
        cache_hit  = 0
        for hdf5_key in self.key2entry:
            number_of_entries += 1
            entry = self.key2entry[hdf5_key]
            if npz_name in entry.requesters:
                if entry.hdf5_in_progress:
                    cache_hit_in_progress += 1
                elif entry.vector is None:
                    need_hdf5.append(entry)
                    cache_miss += 1
                else:
                    self.npz2ready_count[npz_name] += 1
                    cache_hit += 1
        workdir = npz_name + '-workdir'
        os.makedirs(workdir)
        self.next_part = 1
        self.p_submit(need_hdf5, workdir)
        print('Elmo cache statistics:')
        print('\t# cache entries:', number_of_entries)
        print('\t# hdf5 tasks:', len(self.hdf5_tasks))

    def p_submit(self, entries, workdir):
        max_per_elmo_task = 10000
        if 'TT_DEBUG' in os.environ:
            max_per_elmo_task = 500
        if len(entries) > max_per_elmo_task:
            middle = int(len(entries)/2)
            part_1 = entries[:middle]
            part_2 = entries[middle:]
            n1 = self.p_submit(part_1, workdir)
            n2 = self.p_submit(part_2, workdir)
            return n1 + n2
        # write sentences to file
        conllu_file = '%s/part-%03d.conllu' %(workdir, self.next_part)
        hdf5_name   = '%s/part-%03d.hdf5'   %(workdir, self.next_part)
        self.next_part += 1
        try:
            # Python 3
            _file = open(conllu_file, mode='w', encoding='utf-8')
        except TypeError:
            # Python 2
            _file = open(conllu_file, mode='wb')
        lcode = None
        for entry in entries:
            for token in entry.tokens:
                row = []
                row.append(token)
                for i in range(9):
                    row.append('_')
                row = '\t'.join(row)
                _file.write(row)
                _file.write('\n')
            _file.write('\n')
            if not lcode:
                lcode = entry.lcode
            elif lcode != entry.lcode:
                raise ValueError('Inconsistent lcode (%s, %s) for %s' %(
                    lcode, entry.lcode, conllu_file
                ))
        _file.close()
        # submit elmo hdf5 task
        command = []
        command.append('./get-elmo-vectors.sh')
        command.append(conllu_file)
        command.append(lcode)
        command.append(workdir)
        command.append(hdf5_name)
        task = common_udpipe_future.Task(command, 'elmo-hdf5')
        task.submit()
        task.entries = entries
        task.conllu_file = conllu_file
        task.hdf5_name   = hdf5_name
        self.hdf5_tasks.append(task)
        return 1

    def collect(self, sentence, npz_name):
        _, _, _, hdf5_key = sentence
        entry = self.key2entry[hdf5_key]
        if not entry.vectors:
            raise ValueError('trying to collect vectors for kdf5_key that is not ready')
        entry.requesters.remove(npz_name)  # removes one occurrence only
        return entry.vectors

    def scan_for_ready_vectors(self):
        for task in self.hdf5_tasks:
            if task.is_ready():
                for _, _, _, hdf5_key in task.sentences:
                    vectors = hdf5_data[hdf5_key]
                    item = self.key2npz_names[hdf5_key]
                    self.npz2ready_count[task.npz_name] += 1

def train(
    dataset_filename, seed, model_dir,
    epoch_selection_dataset = None,
    monitoring_datasets = [],
    lcode = None,
    batch_size = 32,
    epochs = 60,
):
    if lcode is None:
        raise ValueError('Missing lcode; use --module-keyword to specify a key-value pair')
    if epoch_selection_dataset:
        raise ValueError('Epoch selection not supported with udpipe-future.')
    command = []
    command.append('./elmo_udpf-train.sh')
    command.append(dataset_filename)
    command.append(lcode)
    if seed is None:
        raise NotImplementedError
    command.append(seed)
    command.append(model_dir)
    command.append('%d' %batch_size)
    command.append(common_udpipe_future.get_training_schedule(epochs))
    for i in range(2):
        if len(monitoring_datasets) > i:
            command.append(monitoring_datasets[i].filename)
    common_udpipe_future.run_command(command)
    if common_udpipe_future.incomplete(model_dir):
        if common_udpipe_future.memory_error(model_dir):
            # do not leave erroneous model behind
            os.rename(model_dir, model_dir+('-oom-%d' %batch_size))
            # try again with smaller batch size:
            if batch_size == 1:
                raise ValueError('Cannot train parser even with batch size 1.')
            new_batch_size = int(batch_size/2)
            print('Parser ran out of memory. Re-trying with batch size %d' %new_batch_size)
            train(dataset_filename, seed, model_dir,
                epoch_selection_dataset = epoch_selection_dataset,
                monitoring_datasets = monitoring_datasets,
                lcode = lcode,
                batch_size = new_batch_size,
            )
        else:
            # do not leave incomplete model behind
            error_name = model_dir + '-incomplete'
            os.rename(model_dir, error_name)
            raise ValueError('Model is missing essential files: ' + error_name)

def predict(model_path, input_path, prediction_output_path):
    command = []
    command.append('./elmo_udpf-predict.sh')
    command.append(model_path)
    command.append(input_path)
    command.append(prediction_output_path)
    common_udpipe_future.run_command(command)

def main():
    last_arg_name = 'QUEUENAME'
    last_arg_description = """
QUEUENAME:

    elmo-npz  for the .npz compiler with elmo cache (recommended
              to run exactly one; each instance needs its own
              EFML_CACHE_DIR),

    elmo-hdf5 for the .hdf5 workers (run as many as needed)"""
    if len(sys.argv) > 1 and sys.argv[-1] == 'elmo-npz':
        cache = ElmoCache()
        common_udpipe_future.main(
            queue_name = 'elmo-npz',
            task_processor = ElmoNpzTask,
            cache = cache,
            last_arg = (last_arg_name, last_arg_description),
        )
    elif len(sys.argv) > 1 and sys.argv[-1] == 'elmo-hdf5':
        common_udpipe_future.main(
            queue_name = 'elmo-hdf5',
            task_processor = common_udpipe_future.Task,
            cache = None,
            last_arg = (last_arg_name, last_arg_description),
        )
    else:
        common_udpipe_future.print_usage(
            last_arg = (last_arg_name, last_arg_description),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

