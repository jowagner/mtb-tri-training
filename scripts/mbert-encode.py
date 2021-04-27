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
opt_expand_to    = 0
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
                            -n = use average of last n layers
                             0 = BERT's embedding layer (context-free)
                             1 = first layer
                            12 = 12th layer, should be identical to -1
                            (default: -1)

    --expand-to  NUMBER     When averaging over k layers, expand the
                            output dimension to NUMBER, inserting into
                            each vector a range of unpopulated components
                            and then only averaging the populated
                            components, which are positioned to reduce
                            the number of values that are averaged.
                            Must not exceed 2 * BERT's output dimension.
                            (Default: 0 = keep dimension as is)

    --pooling  METHOD       How to pool the vectors of tokens that have
                            more than one subword unit. One of "first",
                            "last", "max", "average" and "binomial{d}".
                            The maximum is element-wise. Binomial1234
                            takes a weighted average using the binomial
                            distribution with p = 0.1234 as weights.
                            Tokens are pooled after the layers have been
                            merged if more than one layer is selected.
                            (default: first)

    --batch-size  NUMBER    How many sequences to feed into mBERT in one
                            go
                            (Default: 96)

    --length  NUMBER        Maximum sequence length in subword units to
                            feed into mBERT
                            (default: 512)

    --use-gpu               Move tensors and model to cuda device

    --debug                 Print detailed info
                            (combine with a small value for --length to
                            test handling of long sentences and/or long
                            tokens)

    --progress              Show progress information every second

    --verbose               print basic info and information messages
                            from PyTorch and Transformers
                            (Default: only show warnings and print a
                            summary at the end)

    --quiet                 Do not print the summary at the end

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
        elif option in ('--expand-to', '--expand'):
            opt_expand_to = int(sys.argv[1])
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

if opt_output_layer > -2 and opt_expand_to:
    raise ValueError('need multiple layers to expand output vector')
if opt_expand_to < 0:
    raise ValueError('vector dimension must not be negative')

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

if opt_verbose or opt_debug:
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

def print_progress(sep = '\n', print_time = True):
    row = []
    if print_time:
        row.append((time.ctime(time.time())))
    for key in sorted(list(event_counter.keys())):
        row.append(('%s: %d' %(key, event_counter[key])))
    print(sep.join(row))

def encode_batch(batch):
    log_starting('encode_batch')
    lengths, s_idxs, p_idxs, is_last, hdf5_keys, sequences = zip(*batch)
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
        print_encoded_batch(sequences, encoded_batch, s_idxs, p_idxs, is_last, lengths)
    if opt_use_gpu:
        encoded_batch.to('cuda')
    log_finished('encode_batch')
    return (s_idxs, p_idxs, is_last, hdf5_keys, encoded_batch)

def print_encoded_batch(sequences, encoded_batch, s_idxs, p_idxs, is_last, lengths):
    global tokenizer
    print('batch:')
    for b_index, sequence in enumerate(sequences):
        print('[%d]' %b_index)
        print('\ts_idx:', s_idxs[b_index])
        print('\tp_idx:', p_idxs[b_index])
        print('\tis_last:', is_last[b_index])
        print('\tn_units:', lengths[b_index])
        print('\tsequence:', sequence)
        token_ids = encoded_batch['input_ids'][b_index]
        print('\tsubword units:', tokenizer.convert_ids_to_tokens(token_ids))
        print('\t.word_ids():', encoded_batch.word_ids(batch_index = b_index))

def protect_special(sentence):
    global specials_and_replacements
    global opt_debug
    text = ' '.join(sentence)
    replacement_may_be_required = False
    for special, _ in specials_and_replacements:
        if special in text:
            replacement_may_be_required = True
            break
    for code_point in range(0xFFF0, 0xFFFF + 1):
        if chr(code_point) in text:
            replacement_may_be_required = True
            break
    if not replacement_may_be_required:
        return sentence
    if opt_debug: print('protecting special token(s)')
    new_sentence = []
    for token in sentence:
        for special, replacement in specials_and_replacements:
            token = token.replace(special, replacement)
        # BERT ignors Unicode replacement characters but we must
        # have at least one subword unit for each token
        contains_valid_char = False
        for c in token:
            if ord(c) < 0xFFF0 or ord(c) > 0xFFFF:
                contains_valid_char = True
                break
        if not contains_valid_char:
            token = '[UNK]'
        new_sentence.append(token)
    return new_sentence

def get_hdf5_key(sentence):
    # replicate elmoformanylangs hdf5 key
    text = '\t'.join(sentence)
    text = text.replace('.', '$period$')
    text = text.replace('/', '$backslash$')  # [!sic]
    return text

def get_batches(
    conllu_file, batch_size = None,
    buffer_batches = 8,
    shuffle_buffer = None,
    in_order_of_length = True,
):
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
                    n_subword_units,
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
        if shuffle_buffer:
            random.shuffle(s_buffer)
        if in_order_of_length and len(s_buffer) >= buffer_size:
            n_batches = len(s_buffer) // batch_size
            if n_batches:
                n_selected = batch_size * n_batches
                if opt_debug: print('producing %d sorted batches from %d of %d items' %(n_batches, n_selected, len(s_buffer)))
                selected  = s_buffer[:n_selected]
                remaining = s_buffer[n_selected:]
                selected.sort()
                for _ in range(n_batches):
                    yield encode_batch(selected[:batch_size])
                    selected = selected[batch_size:]
                s_buffer = remaining
        while len(s_buffer) >= buffer_size:
            yield encode_batch(s_buffer[:batch_size])
            s_buffer = s_buffer[batch_size:]
        s_index += 1
        if opt_debug: print('%d part(s) created for sentence %d' %(part_index, s_index))
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

# https://gist.github.com/rougier/ebe734dcc6f4ff450abf
# with suggestions from
# https://gist.github.com/keithbriggs
# and alisianoi's answer on
# https://stackoverflow.com/questions/26560726/python-binomial-coefficient
# (can be replaced with math.comb() when Python 3.8 or
# greater is used widely)
def binomial(n, k):
    # For compatibility with scipy.special.{comb, binom} returns 0 instead.
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    b = 1
    for t in range(min(k, n - k)):
        b = (b * n) // (t + 1)
        n -= 1
    return b

def get_binomial_distribution(n, p):
    q = 1 - p
    pks = [1.0]
    qks = [1.0]
    for _ in range(n+1):
        pks.append(pks[-1]*p)
        qks.append(qks[-1]*q)
    retval = []
    for k in range(n+1):
        retval.append(
            binomial(n, k) * pks[k] * qks[n-k]
        )
    return retval

def pool(vectors, span, pooling_method):
    assert len(span) > 0
    if pooling_method == 'first' or len(span) == 1:
        return vectors[span[0]]
    if pooling_method == 'last':
        return vectors[span[-1]]
    if pooling_method.startswith('max'):
        retval = vectors[span[0]]
        for v_index in span[1:]:
            retval = torch.maximum(retval, vectors[v_index])
        return retval
    if pooling_method == 'average':
        weights = len(span) * [1.0 / len(span)]  # uniform weights
    elif pooling_method.startswith('binomial'):
        p = float('0.'+(pooling_method[8:]))
        weights = get_binomial_distribution(len(span)-1, p)
    # add other weight distributions here
    else:
        raise ValueError('unknown pooling method %s' %pooling_method)
    retval = 0  # auto-expands to tensor of zeros below
    for w_index, v_index in enumerate(span):
        retval += vectors[v_index] * weights[w_index]
    return retval

class LayerWithGap:

    def __init__(self, layer, gap_start, gap_end):
        assert gap_start <= gap_end
        self.layer     = layer
        if gap_start == gap_end:
            gap_start = gap_end = 0
        self.gap_start = gap_start
        self.gap_end   = gap_end
        self.position  = 0
        self.gap_width = gap_end - gap_start
        self.batch_size, self.seq_length, self.h_dim = self.layer.shape

    def get_next(self, width):
        assert width > 0
        if not self.gap_start and not self.gap_end \
        and not self.position \
        and self.h_dim == width:
            # speed up pass through
            self.position = width
            return self.layer
        new_position = self.position + width
        if self.position >= self.gap_end:
            # after gap
            slice_start = self.position - self.gap_width
        elif new_position <= self.gap_start:
            # before gap
            slice_start = self.position
        elif self.position >= self.gap_start \
        and new_position <= self.gap_end:
            # inside gap:
            self.position = new_position
            return None
        else:
            raise ValueError('cannot slice %d-dimensional layer with gap %r from virtual position %d with width %d' %(
                h_dim, (gap_start, gap_end), self.position, width
            ))
        self.position = new_position
        retval = self.layer[..., slice_start:slice_start+width]
        assert retval.shape[0] == self.batch_size
        assert retval.shape[1] == self.seq_length
        assert retval.shape[2] == width
        return retval

def average_and_expand(all_layers, n_layers = 4, expand_to = 0):
    log_starting('average_and_expand')
    # each layer has shape (batch size, sequence length, model hidden dimension)
    if not expand_to:
        expand_to = all_layers[-1].shape[-1]
    positions = set()
    positions.add(0)
    positions.add(expand_to)
    layers_with_gap = []
    for index in range(n_layers):
        layer_index = len(all_layers) - (index+1)
        layer = all_layers[layer_index]
        h_dim = layer.shape[-1]
        gap = expand_to - h_dim
        if gap > 0:
            gap_start = (index * h_dim) // (n_layers - 1)
            gap_end = gap_start + gap
            positions.add(gap_start)
            positions.add(gap_end)
        else:
            gap_start = gap_end = 0
        layers_with_gap.append(LayerWithGap(layer, gap_start, gap_end))
        if opt_debug: print('layer %d with gap %r' %(layer_index, (gap_start, gap_end)))
    parts = []
    last_position = 0
    for position in sorted(list(positions)):
        width = position - last_position
        if width:
            # get slices
            to_be_averaged = []
            for layer_with_gap in layers_with_gap:
                layer_slice = layer_with_gap.get_next(width)
                if layer_slice is not None:
                    to_be_averaged.append(layer_slice)
            if opt_debug: print('averaging %d slices of width %d at position %d' %(len(to_be_averaged), width, last_position))
            # get average
            assert len(to_be_averaged) > 0
            weight = 1.0 / len(to_be_averaged)  # uniform weights
            average = 0  # auto-expands to tensor of zeros below
            for layer_slice in to_be_averaged:
                average += layer_slice * weight
            # add to stack
            parts.append(average)
        last_position = position
    # put them together
    retval = torch.cat(parts, dim = 2)
    assert retval.shape[2] == expand_to
    log_finished('average_and_expand')
    return retval

if opt_output_layer > -2:
    print('Layer:', opt_output_layer)
else:
    print('Layer: average of last', -opt_output_layer)

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
            output_hidden_states = opt_output_layer != -1,
            return_dict = True,
            **encoded_batch
        )
        if opt_output_layer == -1:
            selected_layer = outputs.last_hidden_state
        elif opt_output_layer < -1:
            selected_layer = average_and_expand(
                outputs.hidden_states,
                n_layers  = -opt_output_layer,
                expand_to = opt_expand_to
            )
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
                    if span and word_id != last_word_id:
                        # transition to a new input token
                        assert last_word_id is not None
                        event_counter['token with %03d subword unit(s)' %len(span)] += 1
                        vector = pool(vectors, span, opt_pooling)
                        parts.append(vector)
                        span = []
                    if word_id is not None:
                        span.append(v_index)
                    last_word_id = word_id
            if opt_debug: print('number of vectors:', len(parts))
            vectors = torch.stack(parts, dim = 0)
            log_finished('pooling vectors')
            if opt_debug: print('shape of concatenation:', vectors.shape)
            if opt_debug: print('hdf5_key:', repr(hdf5_key))
            if len(parts) != hdf5_key.count('\t') + 1:
                print('Error: %d parts but %d tabs in hdf5 key %r of sentence %d' %(
                    len(parts), hdf5_key.count('\t'), hdf5_key, s_index + 1
                ))
                sys.exit(1)
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
                elif opt_debug:
                    print('Warning: Either found an sha256 hash collision or')
                    print('the input data contains the hash (and the matching')
                    print('number of "x" tokens) for one of the input')
                    print('sentences. The same list of vectors will be')
                    print('returned for the two conflicting sentences of same')
                    print('length. For sentence %d, this vector representation')
                    print('is likely to be bad.')
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

