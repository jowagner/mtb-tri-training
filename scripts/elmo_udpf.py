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

import array
import base64
import hashlib
import math
import numpy
import os
import subprocess
import sys
import time

import utilities

try:
    import h5py
except:
    # only show error if user needs h5py functionality
    pass

try:
    from StringIO import StringIO
except:
    from io import BytesIO as StringIO

import common_udpipe_future

def request_npz(conllu_filename, npz_filename):
    task = ElmoNpzTask([conllu_filename, npz_filename], 'elmo-npz')
    task.submit()

class ElmoNpzTask(common_udpipe_future.Task):

    def __init__(self, command, **kw_args):
        if 'cache' in kw_args:
            self.cache = kw_args['cache']
            del kw_args['cache']
        common_udpipe_future.Task.__init__(self, command, queue_name = 'elmo-npz', **kw_args)

    def start_processing(self):
        print('Starting elmo-npz task for', self.command)
        conllu_file, npz_name, lcode = self.command
        sentences = self.read_sentences(conllu_file)
        try:
            if lcode != self.lcode:
                print('*** self.lcode pre-defined and not matching command lcode ***')
        except:
            print('self.lcode needs to be initialised in start_processing() as expected')
        self.lcode = lcode
        self.npz_name = npz_name
        print('Registering sentences with elmo cache...')
        for tokens, hdf5_key in sentences:
            self.cache.request(tokens, hdf5_key, npz_name, lcode)
        print('Registered all %d sentence(s) with elmo cache' %len(sentences))
        print('Asking elmo cache to submit required elmo-hdf5 tasks...')
        self.cache.submit(npz_name)
        print('Asked elmo cache to submit required elmo-hdf5 tasks')
        self.sentences = sentences
        self.number_of_sentences = len(sentences)

    def finished_processing(self):
        if self.sentences is None:
            # finished in a previous call
            print('task.sentences indicates that npz task %s has finished and npz file is ready' %self.task_id)
            return True
        _, npz_name, lcode = self.command
        n_ready = self.cache.get_n_ready(npz_name)
        if n_ready < self.number_of_sentences:
            # not ready yet
            print('%d of %d sentences for task %s are ready' %(n_ready, self.number_of_sentences, self.task_id))
            return False
        # all vectors are ready
        print('All vectors for task %s are ready --> creating npz file' %self.task_id)
        sentences = self.sentences
        npz_data = {}
        for index, sentence in enumerate(sentences):
            tokens, hdf5_key = sentence
            vectors = self.cache.collect(tokens, hdf5_key, lcode, npz_name)
            npz_key = 'arr_%d' %index
            npz_data[npz_key] = vectors
        # write npz file
        numpy.savez(utilities.std_string(npz_name) + '.prep.npz', **npz_data)
        os.rename(npz_name + b'.prep.npz', npz_name)
        # clean up and mark as finished
        del self.cache.npz2ready_count[npz_name]
        self.sentences = None
        return True

    def read_sentences(self, conllu_file):
        print('ElmoNpzTask.read_sentences', conllu_file)
        id_column    = 0
        token_column = 1
        sentences = []
        try:
            # Python 3
            _file = open(conllu_file, mode='r', encoding='utf-8')
        except TypeError:
            # Python 2
            print('Warning: elmo-npz worker running with Python 2')
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

    def allocate(self, data_file, key, part_index):
        r_index, _ = self.get_location(data_file, key, part_index, accept_free = True)
        self.record_states[r_index] = ord('p')
        return r_index

    def get_location(self, data_file, key, part_index, accept_free = False):
        # TODO: reduce to returning only r_index if 2nd part never used
        next_probe = 0
        while True:
            h = hashlib.sha512(b'%d:%d:%s' %(
                  next_probe, part_index, key
            )).hexdigest()
            r_index = int(h, 16) % self.n_records
            state = self.record_states[r_index]
            if state == ord('i'):
                return r_index, b'init'
            elif state == ord('f') and accept_free:
                return r_index, b'free'
            elif state == ord('p'):
                # pre-allocated for an earlier part of this key
                # --> cannot
                pass
            elif state == ord('d'):
                # data record: need to check whether key and part match
                if (key, part_index) == self.idx2key_and_part[r_index]:
                    return r_index, b'data'
            else:
                raise ValueError('Unknown record state %d at index %d' %(state, r_index))
            next_probe += 1

    def write_vectors(self, data_file, r_index, key, part_index, n_parts, partial_vectors, lcode, n_tokens):
        data_file.seek(r_index * self.record_size)
        payload = []
        payload.append(b'key %s\n' %key)
        payload.append(b'part %d of %d\n' %(part_index+1, n_parts))
        payload.append(b'length %d\n' %n_tokens)  # overall tokens in this sentence, not just in this record
        payload.append(b'lcode %s\n' %lcode)
        binary_vector = StringIO()
        numpy.save(binary_vector, partial_vectors)
        binary_vector.seek(0)
        binary_vector = binary_vector.read()
        payload.append(base64.b64encode(binary_vector))
        payload.append(b'\n')
        payload = b''.join(payload)
        payload_lines = payload.count(b'\n')
        payload_hash = utilities.bstring(hashlib.sha256(payload).hexdigest())
        header = b'data %d %d %s\n' %(r_index, payload_lines, payload_hash)
        data = b''.join((header, payload, b'\n'))
        pad_to = 4096 * int((len(data)+4095)/4096)
        data_file.write(self.get_data_record(
            r_index,
            prefix = data,
            pad_to = pad_to,
        ))
        self.idx2key_and_part[r_index] = (key, part_index)
        self.record_states[r_index] = ord('d')

    def prune_cache_to_max_load_factor(self):
        print('*** TODO: check load factor and prune cache if necessary ***') # TODO

    def on_worker_idle(self):
        self.sync_to_disk_files()

    def on_worker_exit(self):
        self.sync_to_disk_files(force = True)

    def sync_to_disk_files(self, force = False):
        now = time.time()
        if now < self.last_sync + self.sync_interval and not force:
            return
        self.last_sync = now
        self.prune_cache_to_max_load_factor()
        start_time = time.time()
        data_file = open(self.data_filename, 'r+b')
        atime_file = open(self.atime_filename, 'r+b')
        vectors_per_record = self.vectors_per_record
        n_entries_total = 0
        n_entries_vectors = 0
        n_entries_atime = 0
        n_records_vectors = 0
        n_records_atime = 0
        for key, entry in utilities.iteritems(self.key2entry):
            n_entries_total += 1
            if entry.access_time_synced and entry.vectors_on_disk \
            or entry.vectors is None:
                # nothing to do for this entry
                continue
            n_tokens = entry.vectors.shape[0]
            if not entry.records:
                # find out where the parts are or will be stored
                n_parts = int((n_tokens+vectors_per_record-1)/vectors_per_record)
                for part_index in range(n_parts):
                    r_index = self.allocate(data_file, key, part_index)
                    entry.records.append(r_index)
            else:
                n_parts = len(entry.records)
            # now we know what records to use
            if not entry.vectors_on_disk:
                vectors = entry.vectors
                for part_index, r_index in enumerate(entry.records):
                    partial_vectors = vectors[:vectors_per_record]
                    self.write_vectors(
                        data_file, r_index, key, part_index, n_parts,
                        partial_vectors, entry.lcode, entry.length,
                    )
                    vectors = vectors[vectors_per_record:]
                n_entries_vectors += 1
                n_records_vectors += n_parts
                entry.vectors_on_disk = True
            if not entry.access_time_synced:
                for r_index in entry.records:
                    self.write_atime(atime_file, r_index, self.last_access)
                n_entries_atime += 1
                n_records_atime += n_parts
                entry.access_time_synced = True
        data_file.close()
        atime_file.close()
        duration = time.time() - start_time
        print('Synced elmo cache to disk:')
        print('\t# duration: %.1f seconds' %duration)
        try:
            nan = math.nan
        except AttributeError:
            nan = float('nan')
        print('\t# entries, i.e. sentences, that needed vectors written: %d (of %d, %.2f%%)' %(
            n_entries_vectors, n_entries_total,
            100.0*n_entries_vectors/n_entries_total if n_entries_total else nan
        ))
        print('\t# records, i.e. vector blocks, that were written: %d (of %d, %.2f%%)' %(
            n_records_vectors, self.n_records,
            100.0*n_records_vectors/self.n_records if self.n_records else nan
        ))
        print('\t# entries, i.e. sentences, that needed access time updated: %d (of %d, %.2f%%)' %(
            n_entries_atime, n_entries_total,
            100.0*n_entries_atime/n_entries_total if n_entries_total else nan
        ))
        print('\t# records, i.e. vector blocks, that had access time updated: %d (of %d, %.2f%%)' %(
            n_records_atime, self.n_records,
            100.0*n_records_atime/self.n_records if self.n_records else nan
        ))

    def load_from_disk(self):
        config = open(self.config_filename, 'rb')
        line = config.readline()
        if not line.startswith(b'record_size'):
            raise ValueError('Missing record size in elmo cache config file')
        self.record_size = int(line.split()[1])
        line = config.readline()
        if not line.startswith(b'vectors_per_record'):
            raise ValueError('Missing vectors per record in elmo cache config file')
        self.vectors_per_record = int(line.split()[1])
        config.close()
        d_size = os.path.getsize(self.data_filename)
        self.n_records = int(d_size / self.record_size)
        self.record_states = array.array('B')
        f_zero = open('/dev/zero', 'rb')
        self.record_states.fromfile(f_zero, self.n_records)
        f_zero.close()
        self.idx2key_and_part = self.n_records * [None]
        data_file = open(self.data_filename, 'rb')
        atime_file = open(self.atime_filename, 'rb')
        for r_index in range(self.n_records):
            data_file.seek(r_index * self.record_size)
            line = data_file.readline(self.record_size)
            #          [0]  [1] [2] [3]
            # example: data 123 500 xyz
            #  [0] = 'data' marks this record as containing data
            #  [1] is the r_index (redundant with the start position
            #      of this line but included to help debugging)
            #  [2] is the number of payload lines, not counting
            #      this line and comment lines starting with '#'
            #      that may occur _after_ the obligatory lines below
            #  [3] a hash of the payload to detect partially
            #      written, damaged or inconsistent data
            self.record_states[r_index] = utilities.b_ord(line[0])
            if not line.startswith(b'data'):
                # block is free/recycled or never been used
                continue
            fields = line.split()
            assert fields[1] == (b'%d' %r_index)
            n_payload = int(fields[2])
            assert n_payload < self.record_size
            payload_hash = fields[3]
            payload_lines = []
            while n_payload:
                line = data_file.readline(self.record_size)
                if line.startswith(b'#'):
                    continue
                payload_lines.append(line)
                n_payload -= 1
            if payload_hash != self.get_payload_hash(payload_lines):
                continue
            # line: key 00017:en:ksdjahfhqwefuih
            fields = payload_lines[0].split()
            assert fields[0] == b'key'
            key = fields[1]
            # line: part 3 of 7
            fields = payload_lines[1].split()
            assert fields[0] == b'part'
            part_index = int(fields[1]) - 1
            n_parts = int(fields[3])
            # cache this information
            self.idx2key_and_part[r_index] = (key, part_index)
            # line: length 17
            fields = payload_lines[2].split()
            assert fields[0] == b'length'
            n_tokens = int(fields[1])
            # line: lcode en
            fields = payload_lines[2].split()
            assert fields[0] == b'lcode'
            lcode = fields[1]
            # get time of last access
            atime_file.seek(self.atime_size*r_index)
            line = atime_file.readline()
            atime = float(line.split()[0])
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
                entry.vectors = numpy.concatenate(parts, axis=0)
                entry.last_access = max(entry.access_times)
                entry.access_time_synced = (
                    entry.last_access == min(entry.access_times)
                )
                entry.vectors_on_disk = True
                # clean up
                entry.have_parts = None
                entry.access_times = None
        data_file.close()
        atime_file.close()
        # remove all incomplete in-memory cache entries
        incomplete = []
        n_atime_not_synced = 0
        n_normal = 0
        for key, entry in utilities.iteritems(self.key2entry):
            if entry.have_parts is not None:
                incomplete.append(key)
            elif not entry.access_time_synced:
                n_atime_not_synced += 1
            else:
                n_normal += 1
        for key in incomplete:
            del self.key2entry[key]
        print('Loaded elmo cache from disk:')
        print('\t# normal entries found:', n_normal)
        print('\t# entries with inconsistent acess time:', n_atime_not_synced)
        print('\t# incomplete entries discarded:', len(incomplete))

    def __init__(self):
        self.npz2ready_count = {}
        self.key2entry = {}
        self.hdf5_tasks = []
        self.hdf5_workdir_usage = {}
        cache_dir = os.environ['EFML_CACHE_DIR']
        self.atime_filename = cache_dir + '/access'
        self.data_filename  = cache_dir + '/data'
        self.config_filename  = cache_dir + '/config'
        self.atime_size = 32
        self.max_load_factor = 0.95
        self.last_scan = 0.0
        self.scan_interval = 2.0
        self.last_sync = 0.0
        self.sync_interval = 30.0
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
            for key, entry in utilities.iteritems(self.key2entry):
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
        self.record_states = array.array('B', self.n_records * [ord('i')])
        config = open(self.config_filename, 'wb')
        config.write(b'record_size %d\n' %self.record_size)
        config.write(b'vectors_per_record %d\n' %self.vectors_per_record)
        config.close()
        data_file = open(self.data_filename, 'wb')
        for r_index in range(self.n_records):
            data_file.write(self.get_data_record(r_index))
        data_file.close()
        atime_file = open(self.atime_filename, 'wb')
        for r_index in range(self.n_records):
            atime_file.write(self.get_atime_record(0, r_index))
        atime_file.close()

    def write_atime(self, atime_file, r_index, last_access):
        atime_file.seek(r_index * self.atime_size)
        atime_file.write(self.get_atime_record(last_access, r_index))

    def get_atime_record(self, atime, r_index):
        fields = []
        fields.append(b'%.1f' %atime)
        # optional / comment fields:
        fields.append(b'%d' %r_index)
        state = self.record_states[r_index]
        if ord('a') <= state <= ord('z'):
            fields.append(utilities.bstring(chr(state)))
        elif not state:
            fields.append(b'-')
        else:
            fields.append(b'?')
        fields.append(utilities.bstring(time.ctime(atime)))
        # force final tab
        fields.append(b'')
        while True:
            record = '\t'.join(fields)
            if len(record) <= self.atime_size:
                break
            # cannot fit r_index into record
            if len(fields) < 2:
                break
            del fields[-2]
        if len(record) > self.atime_size:
            # required field doesn't fit --> reduce precision
            record = b'%.0f\t' %atime
        return self.with_padding(record, self.atime_size)

    def with_padding(self, record, target_size):
        ''' note that the last character may be replaced with a newline
        '''
        length = len(record)
        if length > target_size:
            raise ValueError('Record size %d is too small for %d bytes in %r...' %(target_size, length, record[:20]))
        if length == target_size:
            return record[:target_size-1] + b'\n'
        else:
            padding = (target_size - len(record) - 1) * b'.'
            return record + padding + b'\n'

    def get_data_record(self, r_index, prefix = None, pad_to = None):
        if self.record_size < 128 or self.record_size % 128:
            raise ValueError('Elmo cache record size too small or not a multiple of 128')
        rows = []
        if prefix is None:
            row = b'init %d ' %r_index
        else:
            row = prefix + b' '
        rows.append(self.with_padding(row, 128))
        if pad_to is None:
            record_size = self.record_size
        else:
            record_size = min(pad_to, self.record_size)
        for pad_index in range(1, record_size/128):
            row = b'........ padding %d of %d for record # %d ' %(
                pad_index, self.record_size/128-1, r_index
            )
            rows.append(self.with_padding(row, 128))
        return b''.join(rows)

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
            ('T', 1000**4),
            ('G', 1000**3),
            ('M', 1000**2),
            ('K', 1000),
            ('B', 1),
        ]:
            if cache_size.endswith(suffix):
                multiplier = candidate_multiplier
                cache_size = cache_size[:-len(suffix)]
                break
        n_bytes = float(cache_size) * multiplier
        return int(n_bytes/self.record_size)

    def get_cache_key(self, tokens, hdf5_key, lcode):
        part1 = b'%05d' %len(tokens)
        if len(part1) != 5:
            # this limit is very generous, over 99k tokens
            raise ValueError('Cannot handle sentence with %d tokens' %len(tokens))
        part2 = lcode
        part3 = utilities.bstring(utilities.hex2base62(
            hashlib.sha512(utilities.bstring(hdf5_key)).hexdigest(),
            min_length = 60
        ))
        return b':'.join((part1, part2, part3[:60]))

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
        for key, entry in utilities.iteritems(self.key2entry):
            number_of_entries += 1
            if entry.vectors is not None:
                number_of_vectors += entry.length
                if entry.vectors_on_disk:
                    number_on_disk += entry.length
            if npz_name in entry.requesters:
                if entry.hdf5_in_progress:
                    cache_hit_in_progress += 1
                elif entry.vectors is None:
                    need_hdf5.append(entry)
                    cache_miss += 1
                    entry.hdf5_in_progress = True
                else:
                    self.npz2ready_count[npz_name] += 1
                    cache_hit += 1
        print('Statistics for npz request %s:' %npz_name)
        print('\t# cache miss:', cache_miss)
        print('\t# cache hit:', cache_hit)
        print('\t# hdf5 in progress:', cache_hit_in_progress)
        if need_hdf5:
            workdir = npz_name + b'-workdir'
            os.makedirs(workdir)
            self.next_part = 1
            self.p_submit(need_hdf5, workdir, npz_name)
        print('Elmo cache statistics:')
        print('\t# cache entries:', number_of_entries)
        print('\t# stored vectors (=tokens):', number_of_vectors)
        print('\t# on disk vectors (=tokens):', number_on_disk)
        print('\t# running hdf5 tasks:', len(self.hdf5_tasks))
        print('\t# hdf5 workdirs:', len(self.hdf5_workdir_usage))

    def p_submit(self, entries, workdir, npz_name):
        if not entries:
            return 0
        max_per_elmo_task = 5000
        if 'TT_DEBUG' in os.environ:
            max_per_elmo_task = 400
        if len(entries) > max_per_elmo_task:
            middle = int(len(entries)/2)
            part_1 = entries[:middle]
            part_2 = entries[middle:]
            n1 = self.p_submit(part_1, workdir, npz_name)
            n2 = self.p_submit(part_2, workdir, npz_name)
            return n1 + n2
        # write sentences to file
        conllu_file = b'%s/part-%03d.conllu' %(workdir, self.next_part)
        hdf5_basename = b'part-%03d.hdf5'   %self.next_part
        self.next_part += 1
        try:
            self.hdf5_workdir_usage[workdir] += 1
        except KeyError:
            self.hdf5_workdir_usage[workdir] = 1
        try:
            # Python 3
            _file = open(conllu_file, mode='w', encoding='utf-8')
        except TypeError:
            # Python 2
            print('Warning: running elmo-npz worker with Python 2')
            _file = open(conllu_file, mode='wb')
        lcode = None
        for entry in entries:
            for t_index, token in enumerate(entry.tokens):
                row = []
                row.append('%d' %(t_index+1))
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
        if lcode is None:
            raise ValueError('Trying to submit an elmo hdf5 task without lcode, conllu =', conllu_file)
        # submit elmo hdf5 task
        command = []
        command.append('./get-elmo-vectors.sh')
        command.append(conllu_file)
        command.append(lcode)
        command.append(workdir)
        command.append(hdf5_basename)
        task = common_udpipe_future.Task(command, 'elmo-hdf5')
        task.submit()
        task.entries = entries
        task.conllu_file = conllu_file
        task.hdf5_name   = b'%s/%s' %(workdir, hdf5_basename)
        task.npz_name    = npz_name
        task.workdir     = workdir
        print('Submitted elmo-hdf5 task to produce', task.hdf5_name)
        self.hdf5_tasks.append(task)
        return 1

    def collect(self, tokens, hdf5_key, lcode, npz_name):
        key = self.get_cache_key(tokens, hdf5_key, lcode)
        if key not in self.key2entry:
            raise ValueError('trying to collect vectors for hdf5_key that was not previously requested or has been collected previously and the cache entry has been released')
        entry = self.key2entry[key]
        if entry.vectors is None:
            raise ValueError('trying to collect vectors for hdf5_key that is not ready')
        entry.requesters.remove(npz_name)  # removes one occurrence only
        return entry.vectors

    def get_n_ready(self, npz_name):
        self.scan_for_ready_vectors()
        self.sync_to_disk_files()
        return self.npz2ready_count[npz_name]

    def scan_for_ready_vectors(self):
        now = time.time()
        if now < self.last_scan + self.scan_interval:
            return
        self.last_scan = now
        still_not_ready = []
        for task in self.hdf5_tasks:
            if os.path.exists(task.hdf5_name):
                os.unlink(task.conllu_file)
                hdf5_data = h5py.File(task.hdf5_name, 'r')
                print('Reading hdf5 file', task.hdf5_name)
                for entry in task.entries:
                    entry.vectors = hdf5_data[entry.hdf5_key][()]
                    entry.hdf5_in_progress = False
                    entry.hdf5_key = None           # release memory as no longer needed
                    # all requesters, not just task.npz_name, need
                    # to be notified that the vectors are ready
                    for npz_name in entry.requesters:
                        self.npz2ready_count[npz_name] += 1
                hdf5_data.close()
                os.unlink(task.hdf5_name)
                self.hdf5_workdir_usage[task.workdir] -= 1
            else:
                still_not_ready.append(task)
            if self.hdf5_workdir_usage[task.workdir] == 0:
                os.rmdir(task.workdir)
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
        common_udpipe_future.worker(
            queue_name = 'elmo-npz',
            task_processor = ElmoNpzTask,
            extra_kw_parameters = {'cache': cache},
            last_arg = (last_arg_name, last_arg_description),
            callback = cache,
        )
    elif len(sys.argv) > 1 and sys.argv[-1] == 'elmo-hdf5':
        common_udpipe_future.worker(
            queue_name = 'elmo-hdf5',
            task_processor = common_udpipe_future.Task,
            last_arg = (last_arg_name, last_arg_description),
        )
    elif len(sys.argv) == 5 and sys.argv[1] == 'test-npz':
        # undocumented test mode
        conllu_file = sys.argv[2]
        npz_file    = sys.argv[3]
        if not conllu_file.startswith('/') \
        or not npz_file.startswith('/'):
            raise ValueError('Must specify absolute paths to input and output files')
        lcode       = sys.argv[4]
        task = ElmoNpzTask([conllu_file, npz_file, lcode])
        task.submit()
    else:
        common_udpipe_future.print_usage(
            last_arg = (last_arg_name, last_arg_description),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

