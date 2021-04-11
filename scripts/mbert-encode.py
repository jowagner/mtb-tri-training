#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# adapted from https://huggingface.co/transformers/quickstart.html

import time
script_start = time.time()

from   collections import defaultdict
import h5py
import hashlib
import logging
import os
import random
import sys
import torch
from   torch.distributions.binomial import Binomial
from   transformers import BertTokenizerFast, BertModel

import conllu_dataset

assert sys.version_info[0] >= 3

opt_verbose = False
opt_progress = False
opt_debug   = False
opt_quiet   = False
opt_use_gpu = False
opt_max_sequence_length = 512
opt_input_format = 'auto-detect'
opt_output_layer = -1
opt_batch_size = 96
opt_pooling = 'first'   # one of first, last, max and average
opt_shuffle = False
opt_help    = False

def print_usage():
    print('Usage: %s [options] INFILE OUTFILE' %(os.path.split(sys.argv[0])[-1]))
    print("""
Options:

    --input-format  FORMAT  conll, line or auto-detect
                            (default: auto-detect via filename extension
                            and line for stdin)

    --output-layer  LAYER   which layer to use
                            -1 = use .last_hidden_state
                             0 = BERT's embedding layer (context-free)
                             1 = first layer
                            12 = 12th layer, should be identical to -1
                            (default: -1)

    --pooling  METHOD       first, last, max, average or binomial1234 for
                            binomial weight distribution with p = 0.1234
                            (default: first)

    --batch-size  NUMBER    How many sequences to feed into mBERT in one
                            go
                            (Default: 64)

    --length  NUMBER        Maximum sequence length in subword units to
                            feed into mBERT
                            (default: 512)

    --use-gpu               Move tensors and model to cuda device

    --debug                 Print detailed info
                            (combine with a small value for --length to
                            test handling of long sentences and/or long
                            tokens)

    --progress              Show progress information every second

    --verbose               print basic info
                            (Default: only show warnings)

    --quiet                 Do not even print the duration summary at the
                            end

    --help                  show this message

""")

specials_and_replacements = [
    # in the unlikely event that one of the special tokens appears in input,
    # we replace it with a similar token
    ('[CLS]',  '[CLS2]'),
    ('[SEP]',  '[SEP2]'),
    ('[UNK]',  '[UNK2]'),
    ('[MASK]', '[MASK2]'),
]

if True:
    # process command line options
    while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
        option = sys.argv[1]
        option = option.replace('_', '-')
        del sys.argv[1]
        if option in ('--help', '-h'):
            opt_help = True
            break
        elif option in ('--input-format', '--input'):
            opt_input_format = sys.argv[1]
            del sys.argv[1]
        elif option in ('--output-layer', '--layer'):
            opt_output_layer = int(sys.argv[1])
            del sys.argv[1]
        elif option in ('--pooling', '--pool'):
            opt_pooling = sys.argv[1]
            del sys.argv[1]
        elif option == '--batch-size':
            opt_batch_size = int(sys.argv[1])
            del sys.argv[1]
        elif option in ('--length', '--max-sequence-length'):
            opt_max_sequence_length = int(sys.argv[1])
            del sys.argv[1]
        elif option in ('--use-gpu', '--gpu'):
            opt_use_gpu = True
        elif option == '--debug':
            opt_debug = True
            opt_max_sequence_length = 10
            opt_batch_size = 4
        elif option == '--shuffle':
            opt_shuffle = True
        elif option == '--verbose':
            opt_verbose = True
        elif option == '--progress':
            opt_progress = True
        elif option == '--quiet':
            opt_quiet = True
        else:
            print('Unsupported option %s' %option)
            opt_help = True
            break
    if len(sys.argv) != 3:
        opt_help = True
    if opt_help:
        print_usage()
        sys.exit(0)

filename = sys.argv[1]
if filename == '-':
    infile = sys.stdin
    opt_input_format = 'line'
else:
    infile = open(filename, 'rt')
    if opt_input_format == 'auto-detect':
        if filename.endswith('.conllu'):
            opt_input_format = 'conll'
        elif filename.endswith('.txt'):
            opt_input_format = 'line'
        else:
            raise ValueError('Cannot autodetect file format')

fout = h5py.File(sys.argv[2], 'w')

if opt_verbose:
    logging.basicConfig(level=logging.INFO)

if opt_debug:
    print(time.ctime(time.time()))
    print('Loading model...')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
model     = BertModel.from_pretrained('bert-base-multilingual-cased')
model.eval()  # evaluation mode without dropout

if opt_use_gpu:
    model.to('cuda')

starttime = {}
duration = defaultdict(lambda: 0)
duration['model ready'] = time.time() - script_start

def log_starting(key):
    starttime[key] = time.time()

def log_finished(key):
    if key not in starttime:
        print('Warning: finished', key, 'but start time not recorded')
        return
    duration[key] += (time.time() - starttime[key])
    del starttime[key]

def print_durations():
    print('running: %1.f seconds' %(time.time() - script_start))
    for key in sorted(list(duration.keys())):
        print('\t%s\t%.1f seconds' %(key, duration[key]))

event_counter = defaultdict(lambda: 0)

def print_progress(sep = '\t', print_time = True):
    row = []
    if print_time:
        row.append((time.ctime(time.time())))
    for key in sorted(list(event_counter.keys())):
        row.append(('%s: %d' %(key, event_counter[key])))
    print(sep.join(row))

def encode_batch(batch):
    log_starting('encode_batch')
    s_idxs, p_idxs, is_last, hdf5_keys, sequences = zip(*batch)
    if opt_debug:
        print('creating batch for %d sequences:' %len(batch))
        for index, sequence in enumerate(sequences):
            print('\t[%d]\t%r' %(index, sequence))
    log_starting('tokenize')
    encoded_batch = tokenizer(
        list(sequences), # pre-tokenised input
        is_split_into_words = True,
        return_tensors      = 'pt',
        #add_special_tokens  = True,  # TODO: no effect on output and not clear from documentation what it does
        padding             = True,
    )
    event_counter['sequence encoded'] += len(batch)
    log_finished('tokenize')
    if opt_debug:
        print_encoded_batch(sequences, encoded_batch, s_idxs, p_idxs, is_last)
    if opt_use_gpu:
        encoded_batch.to('cuda')
    log_finished('encode_batch')
    return (s_idxs, p_idxs, is_last, hdf5_keys, encoded_batch)

def print_encoded_batch(sequences, encoded_batch, s_idxs, p_idxs, is_last):
    global tokenizer
    print('batch:')
    for b_index, sequence in enumerate(sequences):
        print('[%d]' %b_index)
        print('\ts_idx:', s_idxs[b_index])
        print('\tp_idx:', p_idxs[b_index])
        print('\tis_last:', is_last[b_index])
        print('\tsequence:', sequence)
        token_ids = encoded_batch['input_ids'][b_index]
        print('\tsubword units:', tokenizer.convert_ids_to_tokens(token_ids))
        print('\t.word_ids():', encoded_batch.word_ids(batch_index = b_index))

def protect_special(sentence):
    global specials_and_replacements
    global opt_debug
    text = ' '.join(sentence)
    replacement_required = False
    for special, _ in specials_and_replacements:
        if special in text:
            replacement_required = True
            break
    if not replacement_required:
        return sentence
    if opt_debug: print('protecting special token(s)')
    new_sentence = []
    for token in sentence:
        for special, replacement in specials_and_replacements:
            token = token.replace(special, replacement)
        new_sentence.append(token)
    return new_sentence

def get_hdf5_key(sentence):
    # replicate elmoformanylangs hdf5 key
    text = '\t'.join(sentence)
    text = text.replace('.', '$period$')
    text = text.replace('/', '$backslash$')  # [!sic]
    return text

def get_batches(conllu_file, batch_size = None, buffer_batches = 2, shuffle_buffer = None):
    global tokenizer
    global opt_debug
    global opt_batch_size
    if not batch_size:
        batch_size = opt_batch_size
    if shuffle_buffer is None:
        shuffle_buffer = opt_shuffle
    s_buffer = []
    buffer_size = batch_size * buffer_batches
    s_index = 0
    for sentence in get_sentences(conllu_file):
        if opt_debug: print('\nsentence %d: %s' %(s_index+1, sentence))
        hdf5_key = get_hdf5_key(sentence)
        sentence = protect_special(sentence)
        # first try to fit the full input into BERT, i.e. set the
        # split point to after the input
        split_point = len(sentence)
        assert split_point > 0
        part_index = 0
        while sentence:
            # test whether this sentence fits into the BERT sequence length
            #if opt_debug: print('trying first %d tokens' %split_point)
            log_starting('tokenize')
            input_token_ids = tokenizer(
                [sentence[:split_point]],
                is_split_into_words = True,
            )['input_ids'][0]
            log_finished('tokenize')
            event_counter['sequence probed'] += 1
            n_subword_units = len(input_token_ids)
            if n_subword_units <= opt_max_sequence_length:
                #if opt_debug: print('%d subword units --> part %d fits ok' %(
                #    n_subword_units, part_index+1
                #))
                left_half = sentence[:split_point]
                right_half = sentence[split_point:]
                is_last_part = not right_half
                s_buffer.append((
                    s_index, part_index, is_last_part, hdf5_key, left_half,
                ))
                part_index += 1
                sentence = right_half
                split_point = len(sentence)
                #if opt_debug and sentence:
                #    print('remaining:', sentence)
            else:
                #if opt_debug: print('%d subword units --> too big' %n_subword_units)
                # TODO: replace inefficient linear search with binary search
                #       (linear search seems fine for data with predominantly short
                #       sentences but a lot of time is spent here when many sentences
                #       are long)
                split_point -= 1
                if split_point == 0:
                    if opt_debug: print('replacing first token with [UNK] to make it fit')
                    # cannot even fit the first token
                    # --> change it to [UNK] and try again
                    # (debugging output confirms that [UNK] is protected)
                    sentence[0] = '[UNK]'
                    split_point = len(sentence)
        while len(s_buffer) >= buffer_size:
            if shuffle_buffer:
                random.shuffle(s_buffer)
            yield encode_batch(s_buffer[:batch_size])
            s_buffer = s_buffer[batch_size:]
        s_index += 1
        if opt_debug: print('%d parts created for sentence %d' %(part_index, s_index))
    if opt_debug: print('finished reading sentences')
    while s_buffer:
        if opt_debug: print('still %d parts in buffer' %len(s_buffer))
        yield encode_batch(s_buffer[:batch_size])
        s_buffer = s_buffer[batch_size:]

def get_sentences(conllu_file):
    global opt_input_format
    if opt_input_format == 'line':
        for line in conllu_file.readlines():
            tokens = line.split()
            if tokens:
                event_counter['sentence read'] += 1
                yield tokens
        return
    assert opt_input_format == 'conll'
    while True:
        log_starting('conllu reading')
        conllu = conllu_dataset.ConlluDataset()
        sentence = conllu.read_sentence(conllu_file)
        log_finished('conllu reading')
        if sentence is None:
            break
        tokens = []
        for item in sentence:
            tokens.append(item[1])
        event_counter['sentence read'] += 1
        yield tokens

def pool(vectors, span, pooling_method):
    if pooling_method == 'first':
        return vectors[span[0]]
    if pooling_method == 'last':
        return vectors[span[-1]]
    if pooling_method.startswith('max'):
        retval = vectors[span[0]]
        for index in span[1:]:
            retval = torch.maximum(retval, vectors[span[index]])
        return retval
    if pooling_method == 'average':
        weights = len(span) * [1.0 / len(span)]  # uniform weights
    elif pooling_method.startswith('binomial'):
        p = float('0.'+(pooling_method[8:]))
        weights = Binomial(len(span), probs = p).probs()
    # add other weight distributions here
    retval = 0
    for w_index, v_index in enumerate(span):
        retval += vectors[index] * weights[w_index]
    return retval

print('Layer:', opt_output_layer)

data = []
s2n_parts = {}
sp2vectors = {}
sp2word_ids = {}
ignore = set()
last_progress = 0.0
with torch.no_grad():
    for s_idxs, p_idxs, is_lasts, hdf5_keys, \
    encoded_batch in get_batches(infile):
        log_starting('apply model')
        outputs = model(
            output_hidden_states = opt_output_layer >= 0,
            return_dict = True,
            **encoded_batch
        )
        if opt_output_layer == -1:
            selected_layer = outputs.last_hidden_state
        else:
            selected_layer = outputs.hidden_states[opt_output_layer]
        log_finished('apply model')
        # this should have shape (batch size, sequence length, model hidden dimension)
        for index, vectors in enumerate(selected_layer):
            hdf5_key = hdf5_keys[index]
            is_last = is_lasts[index]
            if hdf5_key in ignore:
                # the elmoformanylangs hdf5 format does not allow duplicate sentences
                if is_last:  # count this sentence only once
                    event_counter['sentence completed'] += 1
                continue
            s_index = s_idxs[index]
            p_index = p_idxs[index]
            if is_last:
                s2n_parts[s_index] = p_index + 1  # record number of parts
            key = (s_index, p_index)
            sp2vectors[key] = vectors
            sp2word_ids[key] = encoded_batch.word_ids(batch_index = index)
            if opt_debug:
                print('s2n_parts:', s2n_parts)
            # do we have all parts for this sentence?
            if s_index not in s2n_parts:
                # no as we haven't seen the last part yet
                if opt_debug:
                    print('still missing last part of sentence', s_index+1)
                continue
            expected_n_parts = s2n_parts[s_index]
            assert expected_n_parts > 0
            found_all = True
            for p_index in range(expected_n_parts-1):  # last one already tested
                key = (s_index, p_index)
                if key not in sp2vectors:
                    found_all = False
                    break
            if not found_all:
                # some other part is missing
                if opt_debug:
                    print('still missing some part(s) of sentence', s_index+1)
                continue
            # all parts ready
            if opt_debug or opt_verbose:
                print('all parts ready for sentence', s_index+1)
            log_starting('pooling vectors')
            parts = []
            for p_index in range(expected_n_parts):
                vectors  = sp2vectors[(s_index, p_index)]
                if opt_debug: print('shape of vectors of part %d: %s' %(p_index, vectors.shape))
                word_ids = sp2word_ids[(s_index, p_index)]
                last_word_id = None
                span = []
                for v_index, word_id in enumerate(word_ids):
                    if word_id != last_word_id:
                        # transition to a new input token
                        if span:
                            assert last_word_id is not None
                            vector = pool(vectors, span, opt_pooling)
                            parts.append(vector)
                            span = []
                        last_word_id = word_id
                        if word_id is not None:
                            span.append(v_index)
            if opt_debug: print('number of vectors:', len(parts))
            vectors = torch.stack(parts, dim = 0)
            log_finished('pooling vectors')
            if opt_debug: print('shape of concatenation:', vectors.shape)
            if opt_debug: print('hdf5_key:', repr(hdf5_key))
            assert len(parts) == hdf5_key.count('\t') + 1
            if opt_use_gpu:
                # copy the tensor to host memory
                log_starting('GPU copy')
                vectors = torch.Tensor.cpu(vectors)
                log_finished('GPU copy')
            # add to hdf5
            log_starting('writing hdf5')
            try:
                # following line adpated from
                # https://github.com/HIT-SCIR/ELMoForManyLangs/blob/master/elmoformanylangs/__main__.py
                fout.create_dataset(
                    hdf5_key, vectors.shape, dtype = 'float32', data = vectors
                )
            except ValueError:  # something wrong with above key
                old_hdf5_key = hdf5_key
                hdf5_key = hashlib.sha256(
                    hdf5_key.encode('UTF-8')
                ).hexdigest()
                # keep same number of tabs as in old key
                hdf5_key = hdf5_key + old_hdf5_key.count('\t') * '\tx'
                print('Warning: cannot use hdf5_key %r, changing it to %r' %(
                    old_hdf5_key, hdf5_key
                ))
                if hdf5_key not in ignore:  # hdf5 does not allow the same key twice
                    fout.create_dataset(
                        hdf5_key, vectors.shape, dtype = 'float32', data = vectors
                    )
                # ignore sentences with this key in the future
                ignore.add(old_hdf5_key)
            log_finished('writing hdf5')
            # release memory
            for p_index in range(expected_n_parts):
                key = (s_index, p_index)
                del sp2vectors[key]
                del sp2word_ids[key]
            del s2n_parts[s_index]
            ignore.add(hdf5_key)
            event_counter['sentence completed'] += 1
        if opt_debug or (opt_progress and time.time() > last_progress + 1.0):
            print_progress()
            last_progress = time.time()
        if opt_debug or opt_verbose:
            print_durations()
        if opt_debug:
            print('finished batch')
            print()

fout.close()

if filename != '-':
    infile.close()

if opt_progress:
    print_progress()

if not opt_quiet:
    if not opt_progress:
        print_progress(sep = '\n', print_time = False)
    print_durations()

