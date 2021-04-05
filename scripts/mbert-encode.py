#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# adapted from https://huggingface.co/transformers/quickstart.html

import logging
import torch
from   transformers import BertTokenizer, BertModel

import conllu_dataset

opt_verbose = True
opt_debug   = True
opt_use_gpu = False
opt_max_sequence_length = 512

if opt_debug:
    opt_max_sequence_length = 10

if opt_verbose:
    logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model     = BertModel.from_pretrained('bert-base-multilingual-cased')
model.eval()  # evaluation mode without dropout

if opt_use_gpu:
    model.to('cuda')

def encode_batch(batch):
    if opt_debug: print('creating batch for %d sequences' %len(batch))
    s_idxs, p_idxs, is_last, sentences = zip(*batch)
    encoded_batch = tokenizer(
        sentences, # pre-tokenised input
        is_split_into_words = True,
    )
    encoded_batch['sentence_indices'] = s_idxs
    encoded_batch['part_indices']     = p_idxs
    encoded_batch['is_last']          = is_last
    if opt_debug: print_encoded_batch(sentences, encoded_batch)
    if opt_use_gpu:
        encoded_batch.to('cuda')
    return encoded_batch

def print_encoded_batch(sentences, encoded_batch):
    global tokenizer
    print('batch:')
    for b_index, sentence in sentences:
        print('[%d]' %b_index)
        print('\ts_idx:', encoded_batch['sentence_indices'][b_index])
        print('\tp_idx:', encoded_batch['part_indices'][b_index])
        print('\tis_last:', encoded_batch['is_last'][b_index])
        print('\tsentence:', sentences[b_index])
        token_ids = encoded_bactch['input_ids'][b_index]
        print('\tsubword units:', tokenizer.convert_ids_to_tokens(token_ids))
        print('\t.word_ids():', encoded_bactch.word_ids(batch_index = i))


def get_batches(conllu_file, batch_size = 64, buffer_batches = 2, shuffle_buffer = False):
    global tokenizer
    global opt_debug
    s_buffer = []
    buffer_size = batch_size * buffer_batches
    s_index = 0
    for sentence in get_sentences(conllu_file):
        if opt_debug: print('\nsentence %d: %s' %(s_index+1, sentence))
        # first try to fit the full input into BERT, i.e. set the
        # split point to after the input
        split_point = len(sentence)
        assert split_point > 0
        part_index = 0
        while sentence:
            # test whether this sentence fits into the BERT sequence length
            if opt_debug: print('trying first %d tokens' %split_point)
            input_token_ids = tokenizer(
                [sentence[:split_point]],
                is_split_into_words = True,
            )['input_ids'][0]
            n_subword_units = len(input_token_ids)
            if n_subword_units <= opt_max_sequence_length:
                if opt_debug: print('%d subword units --> part %d fits ok' %(
                    n_subword_units, part_index+1
                ))
                left_half = sentence[:split_point]
                right_half = sentence[split_point:]
                is_last_part = not right_half
                s_buffer.append((s_index, part_index, is_last_part, left_half))
                part_index += 1
                sentence = right_half
                split_point = len(sentence)
                if opt_debug and sentence:
                    print('remaining:', sentence)
            else:
                if opt_debug: print('%d subword units --> too big' %n_subword_units)
                # TODO: replace inefficient linear search with binary search
                #       if we spend too much time in this function
                split_point -= 1
                if split_point == 0:
                    if opt_debug: print('replacing first token with [UNK] to make it fit')
                    # cannot even fit the first token
                    # --> change it to [UNK] and try again
                    # TODO: check that [UNK] is protected
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
    while True:
        conllu = conllu_dataset.ConlluDataset()
        sentence = conllu.read_sentence(conllu_file)
        if sentence is None:
            break
        tokens = []
        for item in sentence:
            tokens.append(item[1])
        yield tokens

data = []
s2n_parts = {}
sp2vectors = {}
with torch.no_grad():
    for encoded_batch in get_batches(conllu_file):
        outputs = model(encoded_batch)
        last_layer = outputs[0]
        # this should have shape (batch size, sequence length, model hidden dimension)
        for index, vectors in last_layer:
            s_index = encoded_batch['sentence_indices'][index]
            p_index = encoded_batch['part_indices'][index]
            is_last = encoded_batch['is_last'][index]
            if is_last:
                s2n_parts[s_index] = p_index + 1  # record number of parts
            key = (s_index, p_index)
            sp2vectors[key] = vectors
            # do we have all parts for this sentence?
            if s_index not in s2n_parts:
                # no as we haven't seen the last part yet
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
                continue
            # all parts ready
            if opt_debug: print('all parts ready for sentence', s_index+1)
            # TODO: concatenate vectors, reduce to tokens (pick first or pool)
            #       and add to hdf5
            raise NotImplementedError
            for token in example:
                # https://stackoverflow.com/questions/62317723/tokens-to-words-mapping-in-the-tokenizer-decode-step-huggingface
                print('%18s --> %r' %(
                    token,
                    tokenizer.encode(
                        token,
                        add_special_tokens = False,
                    )
                ))
            # release memory
            for p_index in range(expected_n_parts):
                key = (s_index, p_index)
                del sp2vectors[key]
            del s2n_parts[s_index]

# TODO: write efml-like hdf5 file
raise NotImplementedError

