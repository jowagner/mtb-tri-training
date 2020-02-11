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

try:
    import h5py
except:
    # only show error if user needs h5py functionality
    pass

import common_udpipe_future

def request_npz(conllu_filename, npz_filename):
    task = ElmoNpzTask([conllu_filename, npz_filename], 'elmo-npz')
    task.submit()

class ElmoNpzTask(common_udpipe_future.Task):

    def __init__(self, command, **kw_args):
        if 'cache' in kw_args:
            self.cache = kw_args['cache']
            del kw_args['cache']
        common_udpipe_future.Task.__init__(self, command, **kw_args)

    def start_processing(self):
        conllu_file, npz_name = self.command
        sentences = self.read_sentences(conllu_file)
        lcode = self.lcode
        for tokens, hdf5_key in sentences:
            self.cache.request(tokens, hdf5_key, npz_name, lcode)
        self.cache.submit(npz_name)
        self.sentences = sentences
        self.number_of_sentences = len(sentences)

    def finished_processing(self):
        if self.sentences is None:
            # finished in a previous call
            return True
        _, npz_name = self.command
        if self.cache.get_n_ready(npz_name) < self.number_of_sentences:
            # not ready yet
            return False
        # all vectors are ready
        sentences = self.sentences
        npz_data = {}
        for index, sentence in enumerate(sentences):
            tokens, hdf5_key = sentence
            vectors = self.cache.collect(tokens, hdf5_key, self.lcode, npz_name)
            npz_key = 'arr_%d' %index
            npz_data[npz_key] = vectors
        # write npz file
        numpy.savez(npz_name + '.prep', **npz_data)
        os.rename(npz_name + '.prep', npz_name)
        # clean up and mark as finished
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
                sentences.append((tokens, sent))
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
        def __init__(self, tokens, lcode, hdf5_key):
            self.vectors = None
            self.last_access = 0.0
            self.access_time_synced = False
            self.records            = []       # list of records to sync atime to ([] = not yet allocated)
            self.vectors_on_disk    = False
            self.hdf5_in_progress   = False
            self.requesters         = []       # do not discard entry while there is a requester (= npz_name)
            self.tokens             = tokens   # set to None when hdf5 task is submitted
            self.hdf5_key           = hdf5_key # set to None when vector is collected from hdf5 task
            self.lcode              = lcode
            if tokens is not None:
                self.length = len(tokens)      # only needed for stats
            else:
                self.length = -1

    def load_from_disk(self):
        config = open(self.config_filename, 'rb')
        line = config.readline()
        if not line.startswith('record_size'):
            raise ValueError('Missing record size in elmo cache config file')
        self.record_size = int(line.split()[1])
        line = config.readline()
        if not line.startswith('vectors_per_record'):
            raise ValueError('Missing vectors per record in elmo cache config file')
        self.vectors_per_record = int(line.split()[1])
        config.close()
        d_size = os.path.getsize(self.data_filename)
        self.n_records = int(d_size / self.record_size)
        data_file = open(self.data_filename, 'rb')
        atime_file = open(self.atime_filename, 'rb')
        for r_index in range(self.n_records):
            data_file.seek(r_index * self.record_size)
            line = data_file.readline(self.record_size)
            if not line.startswith('data'):
                continue
            fields = line.split()
            assert fields[1] == ('%d' %r_index)
            n_payload = int(fields[2])
            payload_hash = fields[3]
            payload_lines = []
            while n_payload:
                line = data_file.readline(self.record_size)
                if line.startswith('#'):
                    continue
                payload_lines.append(line)
                n_payload -= 1
            if payload_hash != self.get_payload_hash(payload_lines):
                continue
            # line: key 00017:en:ksdjahfhqwefuih
            fields = payload_lines[0].split()
            assert fields[0] == 'key'
            key = fields[1]
            # line: part 3 of 7
            fields = payload_lines[1].split()
            assert fields[0] == 'part'
            part_index = int(fields[1]) - 1
            n_parts = int(fields[3])
            # line: length 17
            fields = payload_lines[2].split()
            assert fields[0] == 'length'
            n_tokens = int(fields[1])
            # line: lcode en
            fields = payload_lines[2].split()
            assert fields[0] == 'lcode'
            lcode = fields[1]
            # get time of last access
            atime_file.seek(...) # TODO
            line = atime_file.readline()
            atime = float(line)
            # create in-memory cache entry if it does not exist yet
            if not key in self.key2entry:
                entry = ElmoCache.Entry(None, lcode, None)
                entry.length = n_tokens
                entry.have_parts = []
                entry.access_times = []
                entry.n_parts = n_parts
                self.key2entry[key] = entry
            else:
                entry = self.key2entry[key]
                assert entry.n_parts == n_parts
                assert entry.length  == n_tokens
            # add new part to entry
            vectors = self.base64_to_vectors(payload_lines[4:])
            entry.have_parts.append((part_index, vectors, r_index))
            entry.access_times.append(atime)
            if len(entry.have_parts) == entry.n_parts:
                # all parts are ready
                entry.have_parts.sort()
                last_part_index = -1 
                parts = []
                for part_index, vectors, r_index2 in entry.have_parts:
                    assert part_index == last_part_index + 1
                    entry.records.append(r_index2)
                    parts.append(vectors)
                    last_part_index = part_index
                entry.vectors = numpy.stack(parts)  # TODO: check axis
                entry.last_access = max(entry.access_times)
                entry.access_time_synced = (
                    entry.last_access == min(entry.access_times)
                )
                entry.vectors_on_disk = True
                # clean up
                entry.have_parts = None
                entry.access_times = None
        # remove all incomplete in-memory cache entries 
        incomplete = []
        for key, entry in self.key2entry.iteritems():
            if entry.have_parts is not None:
                incomplete.append(key)
        for key in incomplete:
            del self.key2entry[key]

    def __init__(self):
        self.npz2ready_count = {}
        self.key2entry = {}
        self.hdf5_tasks = []
        cache_dir = os.environ['EFML_CACHE_DIR']
        self.atime_filename = cache_dir + '/access'
        self.data_filename  = cache_dir + '/data'
        self.config_filename  = cache_dir + '/config'
        if os.path.exists(self.atime_filename) \
        and os.path.exists(self.data_filename) \
        and os.path.exists(self.config_filename):
            self.load_from_disk()
        else:
            self.record_size = -1
            self.vectors_per_record = -1
            self.n_records = -1
        # check desired configuration against existing cache file
        rewrite_files = False
        if 'EFML_CACHE_RECORD_SIZE' in os.environ:
            record_size = int(os.environ['EFML_CACHE_RECORD_SIZE'])
        else:
            record_size = 32768
        if record_size != self.record_size:
            self.record_size = record_size
            rewrite_files = True
        if 'EFML_CACHE_VECTORS_PER_RECORD' in os.environ:
            vectors_per_record = int(os.environ['EFML_CACHE_VECTORS_PER_RECORD'])
        else:
            vectors_per_record = 16
        if vectors_per_record != self.vectors_per_record:
            self.vectors_per_record = vectors_per_record
            rewrite_files = True
        n_records = self.get_n_records()
        if n_records != self.n_records:
            self.n_records = n_records
            rewrite_files = True
        if rewrite_files:
            # update disk files
            for key, entry in self.key2entry.iteritems():
                entry.access_time_synced = False
                entry.vectors_on_disk    = False
                entry.records            = []
            self.create_new_disk_files()
            self.sync_to_disk_files()

    def create_new_disk_files(self):
        ''' write cache disk files pre-allocating space
            with empty records, according to
              * self.record_size  and
              * self.n_records
        '''
        


    def get_n_records(self):
        if 'EFML_CACHE_SIZE' in os.environ:
            cache_size = os.environ['EFML_CACHE_SIZE']
        elif 'EFML_MAX_CACHE_ENTRIES' in os.environ:
            print('Missing EFML_CACHE_SIZE; interpreting EFML_MAX_CACHE_ENTRIES as GiB')
            cache_size = os.environ['EFML_MAX_CACHE_ENTRIES'] + 'GiB'
        else:
            raise ValueError('No EFML_CACHE_SIZE configured')
        multiplier = 1
        for suffix, candidate_multiplier in [
            ('TiB', 1024**4),
            ('GiB', 1024**3),
            ('MiB', 1024**2),
            ('KiB', 1024),
            ('TB', 1000**4),
            ('GB', 1000**3),
            ('MB', 1000**2),
            ('KB', 1000),
            ('B', 1),
        ]:
            if cache_size.endswith(suffix):
                multiplier = candidate_multiplier
                cache_size = cache_size[:-len(suffix)]
                break
        n_bytes = float(cache_size) * multiplier
        record_size = self.get_record_size()
        return int(n_bytes/record_size)

    def get_cache_key(self, tokens, hdf5_key, lcode)
        part1 = '%05d' %len(tokens)
        if len(part1) != 5:
            # this limit is very generous, over 99k tokens
            raise ValueError('Cannot handle sentence with %d tokens' %len(tokens))
        part2 = lcode
        part3 = utilities.hex2base62(
            hashlib.sha512(hdf5_key).hexdigest(),
            min_length = 60
        )
        return ':'.join((part1, part2, part3[:60]))

    def get_record_padding(self, index, current_size, target_size):
        missing = target_size - current_size
        full_rows = int(missing / 128)
        partial = missing - full_rows
        rows = []
        if partial:
            rows.append((partial-1)*'.')
            rows.append('\n')
        for r_index in range(full_rows):
            marker = '%d:%d' %(index, r_index)
            padding = 128 - len(marker) - 1
            rows.append(marker)
            rows.append(padding * '.')
            rows.append('\n')
        return ''.join(rows)

    def request(self, tokens, hdf5_key, npz_name, lcode):
        key = self.get_cache_key(tokens, hdf5_key, lcode)
        if not key in self.key2entry:
            self.key2entry[key] = ElmoCache.Entry(tokens, lcode, hdf5_key)
        self.key2entry[key].requesters.append(npz_name)

    def submit(self, npz_name):
        if npz_name in self.npz2ready_count:
            raise ValueError('Duplicate request to create %s' %npz_name)
        self.npz2ready_count[npz_name] = 0
        need_hdf5 = []
        number_of_entries = 0
        number_of_vectors = 0
        number_on_disk    = 0
        cache_hit_in_progress = 0
        cache_miss = 0
        cache_hit  = 0
        for key, entry in self.key2entry.iteritems():
            number_of_entries += 1
            if entry.vectors is not None:
                number_of_vectors += entry.length
                if entry.vectors_on_disk:
                    number_on_disk += entry.length
            if npz_name in entry.requesters:
                if entry.hdf5_in_progress:
                    cache_hit_in_progress += 1
                elif entry.vector is None:
                    need_hdf5.append(entry)
                    cache_miss += 1
                else:
                    self.npz2ready_count[npz_name] += 1
                    cache_hit += 1
        print('Statistics for npz request %s:' %npz_name)
        print('\t# cache miss:', cache_miss)
        print('\t# cache hit:', cache_hit)
        print('\t# hdf5 in progress:', cache_hit_in_progress)
        workdir = npz_name + '-workdir'
        os.makedirs(workdir)
        self.next_part = 1
        self.p_submit(need_hdf5, workdir)
        print('Elmo cache statistics:')
        print('\t# cache entries:', number_of_entries)
        print('\t# stored vectors (=tokens):', number_of_vectors)
        print('\t# on disk vectors (=tokens):', number_on_disk)
        print('\t# running hdf5 tasks:', len(self.hdf5_tasks))

    def p_submit(self, entries, workdir):
        max_per_elmo_task = 5000
        if 'TT_DEBUG' in os.environ:
            max_per_elmo_task = 200
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
            if lcode is None:
                lcode = entry.lcode
            elif lcode != entry.lcode:
                raise ValueError('Inconsistent lcode (%s, %s) for %s' %(
                    lcode, entry.lcode, conllu_file
                ))
            # release memory
            entry.tokens = None
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

    def collect(self, tokens, hdf5_key, lcode, npz_name):
        key = self.get_cache_key(tokens, hdf5_key, lcode)
        entry = self.key2entry[hdf5_key]
        if not entry.vectors:
            raise ValueError('trying to collect vectors for kdf5_key that is not ready')
        entry.requesters.remove(npz_name)  # removes one occurrence only
        return entry.vectors

    def get_n_ready(self, npz_name):
        self.scan_for_ready_vectors()
        return self.npz2ready_count[npz_name]

    def scan_for_ready_vectors(self):
        still_not_ready = []
        for task in self.hdf5_tasks:
            if os.path.exists(task.hdf5_name):
                os.unlink(task.conllu_file)
                hdf5_data = h5py.File(task.hdf5_name, 'r')
                for entry in task.entries:
                    entry.vectors = hdf5_data[entry.hdf5_key][()]
                    entry.hdf5_in_progress = False
                    entry.hdf5_key = None           # release memory as no longer needed
                    self.npz2ready_count[task.npz_name] += 1
                hdf5_data.close()
                os.unlink(task.hdf5_name)
            else:
                still_not_ready.append(task)
        self.hdf5_tasks = still_not_ready

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
            extra_kw_parameters = {'cache': cache},
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

