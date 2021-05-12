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
import bz2
from   collections import defaultdict
import hashlib
import math
import os
import subprocess
import sys
import time
import zlib

import utilities

try:
    import h5py
    import numpy
except:
    # only show error if user needs functionality
    pass

try:
    from StringIO import StringIO
except:
    from io import BytesIO as StringIO

import common_udpipe_future


class NpzTask(common_udpipe_future.Task):

    def __init__(self, command, **kw_args):
        if 'cache' in kw_args:
            self.cache = kw_args['cache']
            del kw_args['cache']
        self.module_name_prefix = __name__.split('_')[0]
        if 'queue_name' not in kw_args:
            kw_args['queue_name'] = 'npz'
        common_udpipe_future.Task.__init__(self, command, **kw_args)

    def start_processing(self):
        print('Starting npz task for', self.command)
        conllu_file, npz_name, embcode = self.command
        sentences = self.read_sentences(conllu_file)
        try:
            assert embcode == self.embcode
        except AttributeError:
            # expected situation is that the following line is necessary
            self.embcode = embcode
        self.npz_name = npz_name
        print('Registering sentences with elmo+mbert cache...')
        for tokens, hdf5_key in sentences:
            self.cache.request(tokens, hdf5_key, npz_name, embcode)
        print('Registered all %d sentence(s) with elmo+mbert cache' %len(sentences))
        print('Asking elmo+mbert cache to submit required hdf5 tasks...')
        self.cache.submit(npz_name)
        print('Asked elmo+mbert cache to submit required hdf5 tasks')
        self.sentences = sentences
        self.number_of_sentences = len(sentences)

    def finished_processing(self):
        if self.sentences is None:
            # finished in a previous call
            print('Warning: repeated call to task.finished_processing() after it finished')
            return True
        _, npz_name, embcode = self.command
        n_ready = self.cache.get_n_ready(npz_name)
        std_task_id = utilities.std_string(self.task_id)
        if n_ready < self.number_of_sentences:
            # not ready yet
            print('%d of %d sentences for task %s are ready' %(
                n_ready, self.number_of_sentences,
                std_task_id
            ))
            return False
        # all vectors are ready
        print('All vectors for task %s are ready --> creating npz file' %std_task_id)
        sentences = self.sentences
        npz_data = {}
        for index, sentence in enumerate(sentences):
            tokens, hdf5_key = sentence
            vectors = self.cache.collect(tokens, hdf5_key, embcode, npz_name)
            npz_key = 'arr_%d' %index
            npz_data[npz_key] = vectors
        # write npz file
        try:
            numpy.savez(utilities.std_string(npz_name) + '.prep.npz', **npz_data)
            self.cache.npz_bytes_written += os.path.getsize(npz_name + b'.prep.npz')
            os.rename(npz_name + b'.prep.npz', npz_name)
        except FileNotFoundError:
            print('Cannot save data. Workdir not found for', npz_name)
        # clean up and mark as finished
        del self.cache.npz2ready_count[npz_name]
        self.sentences = None
        return True

    def read_sentences(self, conllu_file):
        print('NpzTask.read_sentences', conllu_file)
        id_column    = 0
        token_column = 1
        sentences = []
        if conllu_file.endswith(b'.bz2'):
            _file = bz2.BZ2File(conllu_file, 'r')  # caveat: binary / no character encoding
            needs_decoding = True
        else:
            try:
                # Python 3
                _file = open(conllu_file, mode='r', encoding='utf-8')
                needs_decoding = False
            except TypeError:
                # Python 2
                print('Warning: npz worker running with Python 2')
                _file = open(conllu_file, mode='rb')
                needs_decoding = False  # not sure this is right
        tokens = []
        while True:
            line = _file.readline()
            if not line:
                if tokens:
                    raise ValueError('No end-of-sentence marker at end of file %r' %path)
                break
            if needs_decoding:
                line = line.decode('UTF-8')
            if line.isspace():
                # apply same processing as in elmoformanylangs/__main__.py
                sent = '\t'.join(tokens)
                # . and / have special meaning in hdf5 path names
                sent = sent.replace('.', '$period$')
                sent = sent.replace('/', '$backslash$')
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
        def __init__(self, tokens, embcode, hdf5_key):
            self.vectors = None
            self.last_access = 0.0
            self.access_time_synced = False
            self.records            = []       # list of records to sync atime to ([] = not yet allocated)
            self.vectors_on_disk    = False
            self.hdf5_in_progress   = False
            self.requesters         = []       # do not discard entry while there is a requester (= npz_name)
            self.tokens             = tokens   # set to None when hdf5 task is submitted
            self.hdf5_key           = hdf5_key # set to None when vector is collected from hdf5 task
            self.embcode            = embcode
            if tokens is not None:
                self.length = len(tokens)      # only needed for stats
            else:
                self.length = -1
            #
            # Reminder: ElmoCache.__init__() is further below
            #

    def check_limits(self):
        if self.npz_write_limit \
        and self.npz_bytes_written > self.npz_write_limit:
            return 'NPZ write limit reached'
        return None

    def allocate(self, data_file, key, part_index):
        r_index = self.get_location(data_file, key, part_index, accept_free = True)
        self.record_states[r_index] = ord('p')
        return r_index

    def get_location(self, data_file, key, part_index, accept_free = False, max_probes = 999999):
        #print('Probing for %s part %d...' %(key, part_index))
        probe_index = 0
        found_earlier_free = False
        earlier_free_r_index = None
        earlier_free_p_index = None
        while probe_index < max_probes:
            if accept_free and found_earlier_free \
            and probe_index > earlier_free_p_index + 10:
                # See comment below for "state == ord('f')" for why we keep
                # scanning for a few more steps.
                break
            h = hashlib.sha512(b'%d:%d:%s' %(
                  probe_index, part_index, key
            )).hexdigest()
            r_index = int(h, 16) % self.n_records
            state = self.record_states[r_index]
            if state == ord('i'):
                #print('\t%d: record %d is initialised' %(probe_index, r_index))
                if accept_free and found_earlier_free:
                    return earlier_free_r_index
                return r_index
            elif state == ord('f'):
                #print('\t%d: record %d is free' %(probe_index, r_index))
                #return r_index
                # We now keep scanning in case there is a matching entry in a
                # later probe in order to stabilise the positions of entries
                # as they are removed and re-added to the cache.
                # However, to not scan unreasonably long, we note the probe
                # index and, at the start of the loop above, limit how far
                # we scan.
                if accept_free and not found_earlier_free:
                    found_earlier_free = True
                    earlier_free_r_index = r_index
                    earlier_free_p_index = probe_index
            elif state == ord('p'):
                # pre-allocated for an earlier part of this key
                # --> cannot use this record
                #print('\t%d: record %d is pre-allocated' %(probe_index, r_index))
                pass
            elif state == ord('d'):
                # data record: need to check whether key and part match
                if (key, part_index) == self.idx2key_and_part[r_index]:
                    return r_index
                #else:
                #    print('\t%d: record %d has mismatching data %s part %d' %(
                #        (probe_index, r_index,) + self.idx2key_and_part[r_index]
                #    ))
            else:
                raise ValueError('Unknown record state %d at index %d' %(state, r_index))
            probe_index += 1
        if accept_free and found_earlier_free:
            # do not complain about reaching max_probes when we
            # found and accept free records
            return earlier_free_r_index
        raise ValueError('Unable to find space in elmo disk cache.')

    def write_vectors(self, data_file, r_index, key, part_index, n_parts, partial_vectors, embcode, n_tokens):
        data_file.seek(r_index * self.record_size)
        payload = []
        payload.append(b'key %s\n' %key)
        payload.append(b'part %d of %d\n' %(part_index+1, n_parts))
        payload.append(b'length %d\n' %n_tokens)  # overall tokens in this sentence, not just in this record
        payload.append(b'embcode %s\n' %embcode)
        binary_vector = StringIO()
        numpy.save(binary_vector, partial_vectors)
        binary_vector.seek(0)
        binary_vector = zlib.compress(binary_vector.read())
        encoded = base64.b64encode(binary_vector)
        if b'\n' not in encoded:
            while encoded:
                if 960 < len(encoded) < 1920:
                    eat = 12 * (len(encoded) // 24)
                else:
                    eat = 960
                payload.append(encoded[:eat])
                payload.append(b'\n')
                encoded = encoded[eat:]
        else:
            payload.append(encoded)
            payload.append(b'\n')
        payload = b''.join(payload)
        payload_hash = self.get_payload_hash(payload)
        n_payload_lines = payload.count(b'\n')
        header = b'data %d %d %s\n' %(r_index, n_payload_lines, payload_hash)
        data = b''.join((header, payload, b'\n'))
        pad_to = 4096 * int((len(data)+4095+512)/4096)
        data_file.write(self.get_data_record(
            r_index,
            prefix = data,
            pad_to = pad_to,
        ))
        self.idx2key_and_part[r_index] = (key, part_index)
        self.record_states[r_index] = ord('d')

    def payload_lines_to_vectors(self, payload_lines):
        compressed_data = base64.b64decode(b''.join(payload_lines))
        binary_vector = StringIO()
        binary_vector.write(zlib.decompress(compressed_data))
        binary_vector.seek(0)
        return numpy.load(binary_vector)

    def get_payload_hash(self, payload):
        return utilities.bstring(hashlib.sha256(payload).hexdigest())

    def prune_cache_to_max_load_factor(self, now):
        ''' Returns list of keys for entries that should be
            written to disk and frees disk space for all
            remaning entries.
            (Such entries are also removed from the in-memory
            cache if they are not currently requested.)
        '''
        print('Checking load factor of elmo+mbert cache...')
        vectors_per_record = self.vectors_per_record
        candidates = []
        for key, entry in utilities.iteritems(self.key2entry):
            if entry.vectors is None:
                # nothing to do for this entry, e.g. hdf5 in progress
                continue
            if entry.requesters:
                stale_for = 0.0   # high priority item
            else:
                stale_for = now - entry.last_access
            n_tokens = entry.vectors.shape[0]
            n_parts = int((n_tokens+vectors_per_record-1)/vectors_per_record)
            candidates.append((stale_for, key, n_parts))
        candidates.sort()
        candidates.append((None, None, self.n_records+1))   # simplify loop below
        max_records = self.max_load_factor * self.n_records
        n_keys = 0
        n_records_so_far = 0
        while n_records_so_far + candidates[n_keys][2] < max_records:
            n_records_so_far += candidates[n_keys][2]
            n_keys += 1
        print('\t%d entries, in order of recent usage, require %d of %d allowed records' %(
            n_keys, n_records_so_far, max_records
        ))
        del candidates[-1]
        retval = candidates[:n_keys]
        print('\tPruning cache back by %d entries...' %(len(candidates)-n_keys))
        failed = 0
        for _, key, _ in candidates[n_keys:]:
            entry = self.key2entry[key]
            # regardless whether there are requesters, we must make available
            # space on disk
            for r_index in entry.records:
                self.record_states[r_index] = ord('f')
            if entry.requesters:
                failed += 1
                # mark as no longer on disk
                entry.records = []
                entry.vectors_on_disk = False
                entry.access_time_synced = True   # TODO: value should not matter
            else:
                # without requesters, we can simply delete the full entry
                del self.key2entry[key]
        print('\t%d entries only deleted from disk but not from memory due to ongoing requests' %failed)
        return retval

    def on_worker_idle(self):
        self.sync_to_disk_files()

    def on_worker_exit(self):
        if self.memory_only:
            return
        self.sync_to_disk_files(force = True)

    def sync_to_disk_files(self, force = False):
        now = time.time()
        if now < self.last_sync + self.sync_interval and not force:
            return
        keys = self.prune_cache_to_max_load_factor(now)
        if self.memory_only:
            self.last_sync = now
            return
        is_complete = self.p_sync_to_disk_files(
            keys,
            time_limit = 30.0 if not force else None,
        )
        now = time.time()
        if is_complete:
            self.last_sync = now
        else:
            # incomplete sync --> make it so that we only spend a
            # few seconds on other tasks and then continue syncing
            self.last_sync = max(  # ensure value is in range
                self.last_sync,    # [old_last_sync, now]
                min(now, now + 10.0 - self.sync_interval)
            )

    def p_sync_to_disk_files(self, keys, time_limit = None):
        print('Syncing elmo+mbert cache to disk...')
        sys.stdout.flush()
        start_time = time.time()
        data_file = open(self.data_filename, 'r+b')
        atime_file = open(self.atime_filename, 'r+b')
        vectors_per_record = self.vectors_per_record
        n_entries_total = len(keys)
        n_entries_vectors = 0
        n_entries_atime = 0
        n_records_vectors = 0
        n_records_atime = 0
        # get subset of keys that need syncing
        filtered_keys = []
        total_dirty_records = 0
        for _, key, n_parts in keys:
            entry = self.key2entry[key]
            if entry.vectors is not None \
            and (not entry.access_time_synced or not entry.vectors_on_disk):
                filtered_keys.append((key, n_parts))
                total_dirty_records += n_parts
        if time_limit:
            now = time.time()
            percentage = 100.0 * (now - start_time) / time_limit
            if percentage > 50.0:
                print('\t%.0f%% of the available time spent on finding dirty entries' %percentage)
            extra_time = 0.1 * time_limit
            if now + extra_time > start_time + time_limit:
                # less than 5 seconds to time limit
                print('\tadding %.1f seconds to make some sync progress' %extra_time)
                time_limit = extra_time + now - start_time
        progress_info = ElmoCache.ProgressInfo(
            total_dirty_records, self.record_size,
            '\t%(percentage).1f%% of records synced, %(speed)s, %(eta)s',
            verbosity_interval = self.verbosity_interval,
        )
        n_records_synced = 0
        for key, n_parts in filtered_keys:
            now = time.time()
            if time_limit and now > start_time + time_limit:
                print('\taborting sync as time limit has been reached')
                break
            progress_info.update(n_records_synced)
            entry = self.key2entry[key]
            if entry.access_time_synced and entry.vectors_on_disk \
            or entry.vectors is None:
                raise ValueError('wrongly filtered entry')
            if not entry.records:
                # find out where the parts are or will be stored
                for part_index in range(n_parts):
                    r_index = self.allocate(data_file, key, part_index)
                    entry.records.append(r_index)
            elif n_parts != len(entry.records):
                raise ValueError('Incorrect number of records for entry in elmo+mbert cache')
            # now we know what records to use
            if not entry.vectors_on_disk:
                vectors = entry.vectors
                for part_index, r_index in enumerate(entry.records):
                    partial_vectors = vectors[:vectors_per_record]
                    self.write_vectors(
                        data_file, r_index, key, part_index, n_parts,
                        partial_vectors, entry.embcode, entry.length,
                    )
                    vectors = vectors[vectors_per_record:]
                n_entries_vectors += 1
                n_records_vectors += n_parts
                entry.vectors_on_disk = True
            if not entry.access_time_synced:
                for r_index in entry.records:
                    self.write_atime(atime_file, r_index, entry.last_access)
                n_entries_atime += 1
                n_records_atime += n_parts
                entry.access_time_synced = True
            n_records_synced += n_parts
        data_file.close()
        atime_file.close()
        progress_info.update(n_records_synced, force_output = True)
        duration = time.time() - start_time
        print('\t# duration: %.1f seconds' %duration)
        try:
            nan = math.nan
        except AttributeError:
            nan = float('nan')
        print('\t# entries, i.e. sentences, that had vectors written: %d (of %d, %.2f%%)' %(
            n_entries_vectors, n_entries_total,
            100.0*n_entries_vectors/n_entries_total if n_entries_total else nan
        ))
        print('\t# records, i.e. vector blocks, that were written: %d (of %d, %.2f%%)' %(
            n_records_vectors, self.n_records,
            100.0*n_records_vectors/self.n_records if self.n_records else nan
        ))
        print('\t# entries, i.e. sentences, that had access time updated: %d (of %d, %.2f%%)' %(
            n_entries_atime, n_entries_total,
            100.0*n_entries_atime/n_entries_total if n_entries_total else nan
        ))
        print('\t# records, i.e. vector blocks, that had access time updated: %d (of %d, %.2f%%)' %(
            n_records_atime, self.n_records,
            100.0*n_records_atime/self.n_records if self.n_records else nan
        ))
        self.print_cache_stats()
        return total_dirty_records == n_records_synced  # is sync complete?

    def print_cache_stats(self):
        print('\t# npz names:', len(self.npz2ready_count))
        print('\t# running hdf5 tasks:', len(self.hdf5_tasks))
        print('\t# hdf5 workdirs:', len(self.hdf5_workdir_usage))
        total_sentences = self.cache_miss + self.cache_hit + self.cache_hit_in_progress
        if not total_sentences:
            total_sentences = 1.0
        print('\t# cache misses: %d (%.1f%%)' %(
            self.cache_miss, 100.0*self.cache_miss/total_sentences
        ))
        print('\t# cache hits with vectors ready: %d (%.1f%%)' %(
            self.cache_hit, 100.0*self.cache_hit/total_sentences
        ))
        print('\t# cache hits with hdf5 in progress: %d (%.1f%%)' %(
            self.cache_hit_in_progress, 100.0*self.cache_hit_in_progress/total_sentences
        ))
        state2freq = defaultdict(lambda: 0)
        for state in self.record_states:
            state2freq[state] += 1
        for state in sorted(list(state2freq.keys())):
            freq = state2freq[state]
            if state is None:
                print('\t# unknown records:', freq)
            else:
                print('\t# %s records: %d' %(chr(state), freq))
        sys.stdout.flush()

    def print_used_record_size_stats(self, embcode, total_bytes, n_used_records, max_used_size):
        # embcode    GiB    records    average    max_size
        if n_used_records > 0:
            average_size = 100.0 * total_bytes / float(n_used_records * self.record_size)
        else:
            average_size = -1.0
        print('\t%20s\t%7.2f\t%7d\t%6.2f%%\t%6.2f%%' %(
            utilities.std_string(embcode),
            total_bytes / 1024.0**3,
            n_used_records,
            average_size,
            100.0 * max_used_size / float(self.record_size)
        ))

    def load_from_disk(self):
        print('Loading elmo+mbert cache from disk...')
        sys.stdout.flush()
        config = open(self.config_filename, 'rb')
        line = config.readline()
        if not line.startswith(b'record_size'):
            raise ValueError('Missing record size in elmo+mbert cache config file')
        self.record_size = int(line.split()[1])
        line = config.readline()
        if not line.startswith(b'vectors_per_record'):
            raise ValueError('Missing vectors per record in elmo+mbert cache config file')
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
        n_discarded_records = 0
        bytes_so_far = 0
        used_size_stats = defaultdict(lambda: 0)
        embcodes_seen = set()
        # TODO: we want to report the I/O speed but limiting `bytes_so_far`
        #       to the actually read data means that we overestimate the
        #       arrival time as `remaining_bytes` does not account for
        #       that records are not read in full
        progress_info = ElmoCache.ProgressInfo(
            self.n_records, self.record_size,
            '\t%(percentage).1f%% of disk cache records read, %(speed)s, %(eta)s',
            verbosity_interval = self.verbosity_interval,
        )
        for r_index in range(self.n_records):
            # TODO: move sections of code into functions to make this loop more readable
            progress_info.update(
                r_index, is_after = False, bytes_so_far = bytes_so_far
            )
            data_file.seek(r_index * self.record_size)
            line = data_file.readline(self.record_size)
            bytes_so_far += len(line)
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
            record_state = utilities.b_ord(line[0])
            self.record_states[r_index] = record_state
            if record_state != ord('d'):
                # block is free/recycled or has never been used
                continue
            fields = line.split()
            reason_for_skipping = None
            try:
                n_payload = int(fields[2])
            except:
                reason_for_skipping = 'the number of payload lines is missing'
            if len(fields) < 4:
                reason_for_skipping = 'there are too few fields in the record header'
            elif fields[0] != b'data':
                reason_for_skipping = 'the first word is not "data"'
            elif fields[1] != (b'%d' %r_index):
                reason_for_skipping = 'the record index does not match'
            elif n_payload < 0:
                reason_for_skipping = 'the number of payload lines is negative'
            elif n_payload > (self.record_size//72):
                reason_for_skipping = 'the number of payload lines is suspiciously high'
            else:
                payload_hash = fields[3]
            payload_lines = []
            hard_limit = (r_index+5) * self.record_size  # allow somewhat oversized records
            while not reason_for_skipping \
            and n_payload and data_file.tell() < hard_limit:
                line = data_file.readline(self.record_size)
                bytes_so_far += len(line)
                if line.startswith(b'#'):
                    continue
                payload_lines.append(line)
                n_payload -= 1
            if not reason_for_skipping \
            and payload_hash != self.get_payload_hash(b''.join(payload_lines)):
                reason_for_skipping = 'wrong payload hash'
            if reason_for_skipping:
                print('skipping record %d because %s' %(
                    r_index, reason_for_skipping
                ))
                n_discarded_records += 1
                self.record_states[r_index] = ord('f')
                self.idx2key_and_part[r_index] = None
                continue
                # TODO: all the exceptions below should also cause the record
                #       to be skipped rather than failing completely
                #       --> move most of this loop into a `try` block
            # keep some stats on record sizes
            used_size = data_file.tell() - r_index * self.record_size
            used_size_stats['total_bytes'] += used_size
            used_size_stats['n_used_records'] += 1
            if used_size > used_size_stats['max_used_size']:
                used_size_stats['max_used_size'] = used_size
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
            # line: embcode elmo:en
            fields = payload_lines[3].split()
            if fields[0] == b'embcode':
                embcode = fields[1]
            elif fields[0] == b'lcode' \
            and len(fields) == 2 \
            and len(fields[1]) == 2:
                # this looks like an old elmo cache entry
                embcode = b'elmo:' + fields[1]
            else:
                raise ValueError('cannot guess embcode for cache record %d from line %r' %(
                    r_index, payload_linex[3]
                ))
            # keep embcode-specific record size stats
            embcodes_seen.add(embcode)
            used_size_stats[(embcode, 'total_bytes')] += used_size
            used_size_stats[(embcode, 'n_used_records')] += 1
            if used_size > used_size_stats[(embcode, 'max_used_size')]:
                used_size_stats[(embcode, 'max_used_size')] = used_size
            # get time of last access
            atime_file.seek(self.atime_size*r_index)
            line = atime_file.readline()
            bytes_so_far += len(line)
            try:
                atime = float(line.split()[0])
            except:
                raise ValueError('atime for record %d is %r' %(r_index, line))
            # create in-memory cache entry if it does not exist yet
            if not key in self.key2entry:
                entry = ElmoCache.Entry(None, embcode, None)
                entry.length = n_tokens
                entry.have_parts = {}
                entry.access_times = []
                entry.n_parts = n_parts
                self.key2entry[key] = entry
            else:
                entry = self.key2entry[key]
                assert entry.n_parts == n_parts
                assert entry.length  == n_tokens
            assert 0 <= part_index
            assert part_index < entry.n_parts
            if entry.have_parts is None \
            or part_index in entry.have_parts:
                # already collected all parts or this part for this entry
                n_discarded_records += 1
                self.record_states[r_index] = ord('f')
                self.idx2key_and_part[r_index] = None
                continue
            # add new part to entry
            vectors = self.payload_lines_to_vectors(payload_lines[4:])   # TODO
            entry.have_parts[part_index] = (r_index, vectors)
            entry.access_times.append(atime)
            if len(entry.have_parts) == entry.n_parts:
                # all parts are ready
                parts = []
                for part_index in sorted(list(entry.have_parts.keys())):
                    r_index2, vectors = entry.have_parts[part_index]
                    entry.records.append(r_index2)
                    parts.append(vectors)
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
        progress_info.update(
            self.n_records, is_after = False, bytes_so_far = bytes_so_far
        )
        # print record size stats
        print('\t             embcode\t  GiB\trecords\taverage\tmaximum size')
        for embcode in sorted(list(embcodes_seen)):
            self.print_used_record_size_stats(
                embcode,
                used_size_stats[(embcode, 'total_bytes')],
                used_size_stats[(embcode, 'n_used_records')],
                used_size_stats[(embcode, 'max_used_size')],
            )
        self.print_used_record_size_stats(
            'total',
            used_size_stats['total_bytes'],
            used_size_stats['n_used_records'],
            used_size_stats['max_used_size'],
        )
        # remove all incomplete in-memory cache entries
        incomplete = []
        n_atime_not_synced = 0
        n_normal = 0
        for key, entry in utilities.iteritems(self.key2entry):
            if entry.have_parts is not None:
                incomplete.append(key)
                for part_index in entry.have_parts:
                    r_index2, _ = entry.have_parts[part_index]
                    self.record_states[r_index2] = ord('f')
                    # release memory:
                    self.idx2key_and_part[r_index2] = None
            elif not entry.access_time_synced:
                n_atime_not_synced += 1
            else:
                n_normal += 1
        for key in incomplete:
            del self.key2entry[key]
        print('\t# normal entries found:', n_normal)
        print('\t# entries with inconsistent acess time:', n_atime_not_synced)
        print('\t# incomplete entries discarded:', len(incomplete))
        sys.stdout.flush()

    def __init__(self, memory_only = False):
        assert sys.version_info[0] == 3  # code for this class is in python 3
        self.memory_only = memory_only
        self.npz2ready_count = {}
        self.key2entry = {}
        self.hdf5_tasks = []
        self.hdf5_workdir_usage = {}
        if not memory_only:
            cache_dir = os.environ['NPZ_CACHE_DIR']
            self.atime_filename = cache_dir + '/access'
            self.data_filename  = cache_dir + '/data'
            self.config_filename  = cache_dir + '/config'
        self.atime_size = 48
        self.max_load_factor = 0.95
        self.last_scan = 0.0
        self.scan_interval = 20.0
        self.last_sync = 0.0
        self.sync_interval = 900.0
        self.verbosity_interval = 60.0
        if 'NPZ_CACHE_MAX_WRITE' in os.environ:
            self.npz_write_limit = utilities.float_with_suffix(
                os.environ['NPZ_CACHE_MAX_WRITE']
            )
        else:
            self.npz_write_limit = None
        self.npz_bytes_written = 0
        if 'TT_DEBUG' in os.environ \
        and os.environ['TT_DEBUG'].lower() not in ('0', 'false'):
            self.scan_interval = 8.5
            self.sync_interval = 720.0
            self.verbosity_interval = 5.0
        self.cache_hit_in_progress = 0
        self.cache_miss = 0
        self.cache_hit  = 0
        # try loading existing cache
        if not memory_only \
        and os.path.exists(self.atime_filename) \
        and os.path.exists(self.data_filename) \
        and os.path.exists(self.config_filename):
            self.load_from_disk()
        else:
            self.record_size = -1
            self.vectors_per_record = -1
            self.n_records = -1
        # check desired configuration against existing cache file
        rewrite_files = False
        if 'NPZ_CACHE_RECORD_SIZE' in os.environ:
            record_size = int(os.environ['NPZ_CACHE_RECORD_SIZE'])
        else:
            record_size = 32768
        if record_size != self.record_size:
            self.record_size = record_size
            rewrite_files = True
        if 'NPZ_CACHE_VECTORS_PER_RECORD' in os.environ:
            vectors_per_record = int(os.environ['NPZ_CACHE_VECTORS_PER_RECORD'])
        else:
            vectors_per_record = 6
        if vectors_per_record != self.vectors_per_record:
            self.vectors_per_record = vectors_per_record
            rewrite_files = True
        n_records = self.get_n_records()
        if n_records != self.n_records:
            self.n_records = n_records
            rewrite_files = True
        if rewrite_files and not memory_only:
            # update disk files
            # (growing or shrinking the cache as needed, keeping
            # as many records as possible)
            for key, entry in utilities.iteritems(self.key2entry):
                entry.access_time_synced = False
                entry.vectors_on_disk    = False
                entry.records            = []
            self.create_new_disk_files()
            self.sync_to_disk_files()

    class ProgressInfo:

        def __init__(
            self, n_records, record_size, template,
            verbosity_interval = 1.0,
            speed_interval_factor = 5,
            speed_interval = None,
        ):
            self.n_records = n_records
            self.record_size = record_size
            self.template = template
            self.verbosity_interval = verbosity_interval
            if speed_interval:
                self.speed_interval = speed_interval
            else:
                self.speed_interval = speed_interval_factor * verbosity_interval
            self.last_verbose = None
            self.recent_updates = []
            self.start = time.time()
            self.recent_updates.append((self.start, 0))

        def update(self, r_index, is_after = False, bytes_so_far = None, force_output = False):
            if is_after:
                r_index += 1
            now = time.time()
            if not self.last_verbose \
            or force_output \
            or now > self.last_verbose + self.verbosity_interval \
            or r_index == self.n_records:
                if bytes_so_far is None:
                    bytes_so_far = r_index * self.record_size
                if r_index < self.n_records:
                    while len(self.recent_updates) >= 2 \
                    and self.recent_updates[0][0] < now - self.speed_interval:
                        del self.recent_updates[0]
                    while self.recent_updates \
                    and self.recent_updates[-1][1] == bytes_so_far:
                        del self.recent_updates[-1]
                    self.recent_updates.append((now, bytes_so_far))
                    last_update, last_bytes = self.recent_updates[0]
                    duration = now - last_update
                else:
                    duration = now - self.start
                    last_bytes = 0
                remaining_bytes = self.n_records * self.record_size - bytes_so_far
                if duration < 0.01:
                    speed = '??.? MiB/s'
                    eta = 'no ETA'
                    bytes_per_second = -1.0
                    remaining_seconds = -1.0
                    eta_systime = -1.0
                else:
                    bytes_per_second = (bytes_so_far-last_bytes) / duration
                    speed = '%.1f MiB/s' %(bytes_per_second / 1024.0**2)
                    if r_index < self.n_records and bytes_per_second > 0.0:
                        remaining_seconds = remaining_bytes / bytes_per_second
                        eta_systime = now + remaining_seconds
                        eta = 'ETA ' + time.ctime(eta_systime)
                    elif r_index < self.n_records:
                        remaining_seconds = -1.0
                        eta_systime = -1.0
                        eta = 'no ETA'
                    else:
                        remaining_seconds = 0
                        eta_systime = now
                        eta = 'finished ' + time.ctime(now)
                n_samples = len(self.recent_updates)
                n_records = self.n_records
                if n_records:
                    percentage = 100.0 * r_index / n_records
                else:
                    if r_index != 0:
                        print('Error: record index %d for a total of %d records' %(
                            r_index, n_records
                        ))
                    percentage = 100.0
                print(self.template %locals())
                sys.stdout.flush()
                self.last_verbose = now

    def create_new_disk_files(self):
        ''' write cache disk files pre-allocating space
            with empty records, according to
              * self.record_size  and
              * self.n_records
        '''
        print('Allocating elmo+mbert npz disk cache...')
        sys.stdout.flush()
        self.record_states = array.array('B', self.n_records * [ord('i')])
        self.idx2key_and_part = self.n_records * [None]
        # (1) config file
        config = open(self.config_filename, 'wb')
        config.write(b'record_size %d\n' %self.record_size)
        config.write(b'vectors_per_record %d\n' %self.vectors_per_record)
        config.close()
        # (2) data records
        data_file = open(self.data_filename, 'wb')
        progress_info = ElmoCache.ProgressInfo(
            self.n_records, self.record_size,
            '\t%(percentage).1f%% of disk cache data records allocated, %(speed)s, %(eta)s',
            verbosity_interval = self.verbosity_interval,
        )
        for r_index in range(self.n_records):
            data_file.write(self.get_data_record(r_index))
            progress_info.update(r_index, is_after = True)
        data_file.close()
        # (3) access records
        atime_file = open(self.atime_filename, 'wb')
        progress_info = ElmoCache.ProgressInfo(
            self.n_records, self.record_size,
            '\t%(percentage).1f%% of disk cache access records allocated, %(speed)s, %(eta)s',
            verbosity_interval = self.verbosity_interval,
        )
        for r_index in range(self.n_records):
            atime_file.write(self.get_atime_record(0, r_index))
            progress_info.update(r_index, is_after = True)
        atime_file.close()
        print('\tdone')
        sys.stdout.flush()

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
            record = b'\t'.join(fields)
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

    def with_padding(self, record, target_size, optional_msg = None):
        ''' note that the last character may be replaced with a newline
            if it is a tab or space character
        '''
        length = len(record)
        if length > target_size:
            raise ValueError('Record size %d is too small for %d bytes in %r...' %(target_size, length, record[:20]))
        if length == target_size:
            if record[-1] in (b'\t', b' '):
                record = record[:-1] + b'\n'
            return record
        elif optional_msg and length + len(optional_msg) + 1 == target_size:
            # can fit optional message exactly
            return record + optional_msg + b'\n'
        elif optional_msg and length + len(optional_msg) + 2 <= target_size:
            # can fit optional message and padding
            padding = (target_size - length - len(optional_msg) - 2) * b'.'
            return record + optional_msg + b'\n' + padding + b'\n'
        else:
            padding = (target_size - length - 1) * b'.'
            return record + padding + b'\n'

    def get_data_record(self, r_index, prefix = None, pad_to = None):
        if self.record_size < 128 or self.record_size % 128:
            raise ValueError('Elmo cache record size too small or not a multiple of 128')
        rows = []
        if prefix is None:
            row = b'init %d ' %r_index
        elif prefix[-1] == b'\n':
            row = prefix
        else:
            row = prefix + b'\t'
        start_pad_lines = int((len(row)+127)/128)
        first_line_pad_to = 128 * start_pad_lines
        optional_msg = b'using %.2f%% of space' %(
            100.0*len(row) / float(self.record_size)
        )
        rows.append(self.with_padding(row, first_line_pad_to, optional_msg))
        if pad_to is None:
            record_size = self.record_size
        else:
            record_size = min(pad_to, self.record_size)
        for pad_index in range(start_pad_lines, int(record_size/128)):
            row = b'........ padding %d of %d for record # %d %s' %(
                pad_index, self.record_size/128-1, r_index,
                optional_msg,
            )
            rows.append(self.with_padding(row, 128))
        return b''.join(rows)

    def get_n_records(self):
        if 'NPZ_CACHE_SIZE' in os.environ:
            cache_size = os.environ['NPZ_CACHE_SIZE']
        else:
            raise ValueError('No NPZ_CACHE_SIZE configured')
        n_bytes = utilities.float_with_suffix(cache_size)
        return int(n_bytes/self.record_size)

    def get_cache_key(self, tokens, hdf5_key, embcode):
        part1 = b'%05d' %len(tokens)
        if len(part1) != 5:
            # this limit is very generous, over 99k tokens
            raise ValueError('Cannot handle sentence with %d tokens' %len(tokens))
        part2 = embcode
        part3 = utilities.bstring(utilities.hex2base62(
            hashlib.sha512(utilities.bstring(hdf5_key)).hexdigest(),
            min_length = 60
        ))
        return b'/'.join((part1, part2, part3[:60]))

    def request(self, tokens, hdf5_key, npz_name, embcode):
        key = self.get_cache_key(tokens, hdf5_key, embcode)
        if not key in self.key2entry:
            self.key2entry[key] = ElmoCache.Entry(tokens, embcode, hdf5_key)
        self.key2entry[key].requesters.append(npz_name)

    def submit(self, npz_name):
        if npz_name in self.npz2ready_count:
            npz_name = utilities.std_string(npz_name)
            raise ValueError('Duplicate request to create %s' %utilities.std_string(npz_name))
        self.npz2ready_count[npz_name] = 0
        need_hdf5 = []
        number_of_entries = 0
        number_of_vectors = 0
        number_on_disk    = 0
        cache_hit_in_progress = 0
        cache_miss = 0
        cache_hit  = 0
        for key, entry in utilities.iteritems(self.key2entry):
            #number_of_entries += 1
            #if entry.vectors is not None:
            #    number_of_vectors += entry.length
            #    if entry.vectors_on_disk:
            #        number_on_disk += entry.length
            if npz_name in entry.requesters:
                n_requests = entry.requesters.count(npz_name)
                if entry.hdf5_in_progress:
                    cache_hit_in_progress += n_requests
                elif entry.vectors is None:
                    need_hdf5.append(entry)
                    cache_miss += n_requests
                    entry.hdf5_in_progress = True
                else:
                    self.npz2ready_count[npz_name] += n_requests
                    cache_hit += n_requests
        self.cache_hit_in_progress += cache_hit_in_progress
        self.cache_miss += cache_miss
        self.cache_hit  += cache_hit
        print('Statistics for npz request %s:' %npz_name)
        print('\t# cache miss:', cache_miss)
        print('\t# cache hit:', cache_hit)
        print('\t# hdf5 in progress:', cache_hit_in_progress)
        if need_hdf5:
            workdir = npz_name + b'-workdir'
            t_index = 0
            while os.path.exists(workdir):
                workdir = b'%s-%d-workdir' %(npz_name, t_index)
                t_index += 1
            os.makedirs(workdir)
            self.next_part = 1
            self.p_submit(need_hdf5, workdir, npz_name)
        print('\t# running hdf5 tasks:', len(self.hdf5_tasks))
        print('\t# hdf5 workdirs:', len(self.hdf5_workdir_usage))
        self.print_cache_stats()

    def p_submit(self, entries, workdir, npz_name):
        if not entries:
            return 0
        max_per_hdf5_task = 2000
        if entries[0].embcode.startswith(b'mbert'):
            max_per_hdf5_task *= 12
        if 'TT_DEBUG' in os.environ \
        and os.environ['TT_DEBUG'].lower() not in ('0', 'false'):
            # create many small tasks for debugging
            max_per_hdf5_task = 400
        if len(entries) > max_per_hdf5_task:
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
            print('Warning: running npz worker with Python 2')
            _file = open(conllu_file, mode='wb')
        embcode = None
        for entry in entries:
            for t_index, token in enumerate(entry.tokens):
                # write token in conllu format
                row = []
                row.append('%d' %(t_index+1))
                row.append(token)
                for i in range(9):   # empty conllu columns
                    row.append('_')
                row = '\t'.join(row)
                _file.write(row)
                _file.write('\n')
            _file.write('\n')  # sentence ends with empty line in conlly format
            if embcode is None:
                embcode = entry.embcode
            elif embcode != entry.embcode:
                raise ValueError('Inconsistent embcode (%s, %s) for %s' %(
                    embcode, entry.embcode, conllu_file
                ))
            # release memory
            entry.tokens = None
        _file.close()
        if embcode is None:
            raise ValueError('Trying to submit an elmo or mbert hdf5 task without embcode, conllu =', conllu_file)
        # submit elmo hdf5 task
        command = []
        if embcode.startswith(b'mbert:'):
            command.append('./get-mbert-vectors.sh')
            target_queue = 'mbert-hdf5'
        elif embcode.startswith(b'elmo:'):
            command.append('./get-elmo-vectors.sh')
            target_queue = 'elmo-hdf5'
        else:
            raise ValueError('embcode %s' %repr(embcode))
        command.append(conllu_file)
        if embcode.startswith(b'elmo:'):
            lcode = embcode.split(b':')[1]
            command.append(lcode)
        elif embcode.startswith(b'mbert:'):
            _, layer, expand_to, pooling = embcode.split(b':')
            command.append(layer)
            command.append(expand_to)
            command.append(pooling)
        command.append(workdir)
        command.append(hdf5_basename)
        task = common_udpipe_future.Task(command, target_queue)
        task.submit()
        task.entries = entries
        task.conllu_file = conllu_file
        task.hdf5_name   = b'%s/%s' %(workdir, hdf5_basename)
        task.npz_name    = npz_name
        task.workdir     = workdir
        print(
            'Submitted', target_queue, 'task to produce',
            utilities.std_string(task.hdf5_name)
        )
        self.hdf5_tasks.append(task)
        return 1

    def collect(self, tokens, hdf5_key, embcode, npz_name):
        key = self.get_cache_key(tokens, hdf5_key, embcode)
        if key not in self.key2entry:
            raise ValueError('trying to collect vectors for hdf5_key that was not previously requested or has been collected previously and the cache entry has been released')
        entry = self.key2entry[key]
        if entry.vectors is None:
            raise ValueError('trying to collect vectors for hdf5_key that is not ready')
        entry.requesters.remove(npz_name)  # removes one occurrence only
        entry.last_access = time.time()
        entry.access_time_synced = False
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
                print('Reading hdf5 file', utilities.std_string(task.hdf5_name))
                for entry in task.entries:
                    try:
                        entry.vectors = hdf5_data[entry.hdf5_key][()]
                    except KeyError:
                        alternative_hdf5_key = hashlib.sha256(
                            entry.hdf5_key.encode('UTF-8')
                        ).hexdigest()
                        # keep same number of tabs as in old key
                        alternative_hdf5_key = alternative_hdf5_key \
                                           + entry.hdf5_key.count('\t') * '\tx'
                        entry.vectors = hdf5_data[alternative_hdf5_key][()]
                    entry.hdf5_in_progress = False
                    entry.hdf5_key = None           # release memory as no longer needed
                    # all requesters, not just task.npz_name, need
                    # to be notified that the vectors are ready
                    for npz_name in entry.requesters:
                        self.npz2ready_count[npz_name] += 1
                    # must keep list of requesters to prevent entry from being
                    # flushed out before requester has a chance to collect the
                    # vector; new requesters may join but will trigger any new
                    # hdf5 tasks because entry.vectors is set
                hdf5_data.close()
                os.unlink(task.hdf5_name)
                self.hdf5_workdir_usage[task.workdir] -= 1
            else:
                still_not_ready.append(task)
            if self.hdf5_workdir_usage[task.workdir] == 0:
                for statsfile in os.listdir(task.workdir):
                    if statsfile.split(b'.')[-1] not in (b'start', b'end'):
                        continue
                    statspath = b'/'.join((task.workdir, statsfile))
                    if os.path.exists(statspath):
                        os.unlink(statspath)
                try:
                    os.rmdir(task.workdir)
                except:
                    print('Warning: Could not remove folder', task.workdir)
                del self.hdf5_workdir_usage[task.workdir]
        self.hdf5_tasks = still_not_ready

class NpzTaskManager:

    def __init__(self, output_dir, embcode, priority = 50):
        self.tasks = []
        workdir = output_dir + '-npz-workdir'
        if not os.path.exists(workdir):
            common_udpipe_future.my_makedirs(workdir)
        self.workdir = workdir
        self.embcode = embcode
        self.next_index = 1
        self.priority = priority

    def append(self, conllu_file):
        npz_file = '%s/file-%d.npz' %(self.workdir, self.next_index)
        self.next_index += 1
        task = NpzTask(
            [conllu_file, npz_file, self.embcode],
            priority = self.priority
        )
        task.submit()
        task.npz_file = npz_file
        self.tasks.append(task)
        return npz_file

    def get_npz_files(self):
        return [task.npz_file for task in self.tasks]

    def get_last_npz_file(self):
        print('Deprecation warning: Please use return value of NpzTaskManager.append() instead of NpzTaskManager.get_last_npz_file().')
        return self.tasks[-1].npz_file

    def cleanup(self):
        for npz_file in self.get_npz_files():
            if os.path.exists(npz_file):
                print('Cleaning up ' + utilities.std_string(npz_file))
                os.unlink(npz_file)
        for statsfile in os.listdir(self.workdir):
            if statsfile.split('.')[-1] not in ('start', 'end'):
                continue
            statspath = '/'.join((self.workdir, statsfile))
            if os.path.exists(statspath):
                os.unlink(statspath)
        os.rmdir(self.workdir)

def train(
    dataset_filename, seed, model_dir,
    epoch_selection_dataset = None,
    monitoring_datasets = [],
    lcode = None,
    batch_size = 32,
    epochs = 60,
    priority = 50,
    is_multi_treebank = False,
    submit_and_return = False,
    mbert_layer = 9,
    mbert_expand_to = 0,
    mbert_pooling = 'average',
):
    if lcode is None:
        raise ValueError('Missing lcode; use --module-keyword to specify a key-value pair')
    if epoch_selection_dataset:
        raise ValueError('Epoch selection not supported with udpipe-future.')
    if __name__.startswith('elmo_'):
        embcode = 'elmo:%s' %lcode
    elif __name__.startswith('mbert_'):
        embcode = 'mbert:%s:%s:%s' %(mbert_layer, mbert_expand_to, mbert_pooling)
    else:
        raise NotImplementedError
    npz_tasks = NpzTaskManager(model_dir, embcode, priority = priority)
    command = []
    command.append('./elmo_udpf-train.sh')
    command.append(dataset_filename)
    command.append(npz_tasks.append(dataset_filename))
    if seed is None:
        # the wrapper script currently does not support
        # calling the parser without "--seed"
        raise NotImplementedError
    command.append(seed)
    command.append(model_dir)
    command.append('%d' %batch_size)
    command.append(common_udpipe_future.get_training_schedule(epochs))
    command.append(lcode)  # for fasttext; will be stored with the model
    # while the parser training parameters do not change when switching
    # to mbert as these parameters go to the npz task only, we need to
    # store these parameters with the model so that they can be
    # retrieved at prediction time
    command.append(embcode)
    if is_multi_treebank:
        command.append('--extra_input tbemb')
    else:
        command.append('')
    for i in range(2):
        if len(monitoring_datasets) > i:
            conllu_file = monitoring_datasets[i]
            if type(conllu_file) is tuple:
                conllu_file = conllu_file[0]
            if type(conllu_file) is not str:
                conllu_file = conllu_file.filename
            command.append(conllu_file)
            command.append(npz_tasks.append(conllu_file))
    task = common_udpipe_future.run_command(
        command,
        requires = npz_tasks.get_npz_files(),
        priority = 400+priority,
        submit_and_return = submit_and_return,
        cleanup = npz_tasks,
    )
    if submit_and_return:
        return task
    if not os.path.exists(model_dir):
        raise ValueError('Failed to train parser (missing output)')
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
                epochs = epochs,
                priority = priority,
                is_multi_treebank = is_multi_treebank,
                submit_and_return = submit_and_return,
                mbert_layer = mbert_layer,
                mbert_expand_to = mbert_expand_to,
                mbert_pooling = mbert_pooling,
            )
        else:
            # do not leave incomplete model behind
            error_name = model_dir + '-incomplete'
            os.rename(model_dir, error_name)
            raise ValueError('Model is missing essential files: ' + error_name)

def predict(
    model_path, input_path, prediction_output_path,
    priority = 50,
    is_multi_treebank = False,
    submit_and_return = False,
    wait_for_input = False,
    wait_for_model = False,
):
    assert sys.version_info[0] == 2  # code below only works in python 2
    # read embcode from model file
    embcode_filename = '%s/embcode.txt' %model_path
    if os.path.exists(embcode_filename):
        embcode_file = open(embcode_filename, 'rb')
        embcode = embcode_file.readline().rstrip()
        embcode_file.close()
    else:
        # support old model format
        if __name__.startswith('mbert_'):
            raise ValueError('cannot guess embcode for old mbert model; please provide embcode.txt')
        if not __name__.startswith('elmo_'):
            raise ValueError('missing embcode.txt')
        # for elmo, we can fall back to using lcode
        lcode_file = open('%s/elmo-lcode.txt' %model_path, 'rb')
        embcode = 'elmo:' + lcode_file.readline().rstrip()
        lcode_file.close()
    # prepare npz files and parser command
    command = []
    npz_tasks = NpzTaskManager(
        prediction_output_path, embcode, priority = priority,
    )
    command.append('./elmo_udpf-predict.sh')
    command.append(model_path)
    command.append(input_path)
    command.append(npz_tasks.append(input_path))
    command.append(prediction_output_path)
    if is_multi_treebank:
        command.append('--extra_input tbemb')
    else:
        command.append('')
    requires = npz_tasks.get_npz_files()
    if wait_for_input:
        requires.append(input_path)
    if wait_for_model:
        requires.append(model_path)
    common_udpipe_future.run_command(
        command,
        requires = requires,
        priority = priority,
        submit_and_return = submit_and_return,
        cleanup = npz_tasks,
    )
    if submit_and_return:
        return task

def main():
    last_arg_name = 'QUEUENAME'
    last_arg_description = """
QUEUENAME:

    npz       for the .npz compiler with elmo+mbert cache (recommended
              to run exactly one; each instance needs its own
              NPZ_CACHE_DIR),

    elmo-hdf5 for the elmo hdf5 workers (run as many as needed)

    mbert-hdf5 for the mbert hdf5 workers (run as many as needed)"""
    if len(sys.argv) > 1 and sys.argv[-1].endswith('npz'):
        queue_name = 'npz'
        del sys.argv[-1]
        try:
            opt_index = sys.argv.index('--memory-only')
            del sys.argv[opt_index]
            memory_only = True
        except ValueError:
            memory_only = False
        cache = ElmoCache(memory_only=memory_only)
        common_udpipe_future.main(
            queue_name = queue_name,
            task_processor = NpzTask,
            extra_kw_parameters = {'cache': cache},
            last_arg = (last_arg_name, last_arg_description),
            callback = cache,
        )
    elif len(sys.argv) > 1 and sys.argv[-1].endswith('-hdf5'):
        queue_name = sys.argv[-1]
        del sys.argv[-1]
        common_udpipe_future.main(
            queue_name = queue_name,
            task_processor = common_udpipe_future.Task,
            last_arg = (last_arg_name, last_arg_description),
        )
    elif len(sys.argv) == 5 and sys.argv[1] == 'test':
        # undocumented test mode: get npz for given conllu file
        conllu_file = sys.argv[2]
        npz_file    = sys.argv[3]
        if not conllu_file.startswith('/') \
        or not npz_file.startswith('/'):
            raise ValueError('Must specify absolute paths to input and output files')
        embcode       = sys.argv[4]
        task = NpzTask([conllu_file, npz_file, embcode])
        task.submit()
    else:
        common_udpipe_future.print_usage(
            last_arg = (last_arg_name, last_arg_description),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

