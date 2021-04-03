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
opt_use_gpu = False
opt_max_sequence_length = 512

if opt_verbose:
    logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model     = BertModel.from_pretrained('bert-base-multilingual-cased')
model.eval()  # evaluation mode without dropout

if opt_use_gpu:
    model.to('cuda')

def get_batch(sentences):
    encoded_batch = tokenizer(
        sentences, # pre-tokenised input
        is_split_into_words = True,
    )
    if opt_use_gpu:
        encoded_batch.to('cuda')
    return encoded_batch

def get_batches(conllu_file, batch_size = 64, buffer_batches = 2, shuffle_buffer = False):
    s_buffer = []
    buffer_size = batch_size * buffer_batches
    s_index = 0
    for sentence in get_sentences(conllu_file):
        split_point = len(sentence)
        assert split_point > 0
        while sentence:
            # test whether this sentence fits into the BERT sequence length
            input_token_ids = tokenizer(
                [sentence[:split_point]],
                is_split_into_words = True,
            )['input_ids'][0]
            if len(input_token_ids) <= opt_max_sequence_length:
                s_buffer.append((s_index, 0, 1, sentence))
                sentence = sentence[split_point:]
            else:
                split_point -= 1
                if split_point == 0:
                    # cannot even fit the first token
                    # --> change it to [UNK] and try again
                    # TODO: check that [UNK] is protected
                    sentence[0] = '[UNK]'
                    split_point = len(sentence)
        while len(s_buffer) >= buffer_size:
            if shuffle_buffer:
                random.shuffle(s_buffer)
            yield get_batch(s_buffer[:batch_size])
            s_buffer = s_buffer[batch_size:]
        s_index += 1
    while s_buffer:
        yield get_batch(s_buffer[:batch_size])
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

# Predict hidden states features for each layer
with torch.no_grad():
    for batch in get_batches(conllu_file):  
        outputs = model(batch)
        last_layer = outputs[0]
        # this should have shape (batch size, sequence length, model hidden dimension)

