#!/usr/bin/env python2
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
import random
import sys
import time

import basic_dataset
import conllu_dataset

filename_extension = conllu_dataset.get_filename_extension()

def get_treebanks_and_minrounds(workdirs):
    # TODO: Scan workdirs to support all languages
    #       for which there is output:
    #        * short_lcode is the first letter of the workdir
    #        * tbid can be found in the prediction filenames (or
    #          the test set symlink in any model)
    #        * lcode can be derived from tbid
    #        * long_tbname can be read from the test set symlink
    #          in any model
    #        * tbname can be derived from long_tbname
    #        * language can be derived from tbname
    #        * minrounds is the minimum number of rounds available
    # For the moment, the list is hard-coded below:
    retval = []
    for (short_lcode, tbid, long_tbname) in [
        ('e', 'en_ewt',    'UD_English-EWT'),
        ('h', 'hu_szeged', 'UD_Hungarian-Szeged'),
        ('u', 'ug_udt',    'UD_Uyghur-UDT'),
        ('v', 'vi_vtb',    'UD_Vietnamese-VTB'),
    ]:
        if not long_tbname.startswith('UD_'):
            raise ValueError('unsupported long_tbname %r' %(long_tbname,))
        lcode = tbid.split('_')[0]
        tbname = long_tbname[3:]
        language = tbname.split('-')[0]
        retval.append((short_lcode, lcode, language, tbid, tbname, long_tbname))
    return retval, 5

def get_treebank_dir(long_tbname):
    # TODO: support other ways to set UD_TREEBANK_DIR
    return '%s/%s' %(os.environ['UD_TREEBANK_DIR'], long_tbname)

def get_vocab_and_morph_and_labels_and_sent_ids(treebank, test_type):
    _, _, _, tbid, _, long_tbname = treebank
    treebank_dir = get_treebank_dir(long_tbname)
    # Collect tokens, morphology and labels from training data
    dataset = conllu_dataset.ConlluDataset()
    f = open('%s/%s-ud-train.conllu' %(treebank_dir, tbid), 'rb')
    dataset.load_file(f)
    f.close()
    vocab = set()
    morph = {}
    labels = set()
    sent_ids = set()
    for sentence in dataset:
        for row in sentence:
            token = row[1]
            vocab.add(token)
            if not token in morph:
                morph[token] = set()
            morph[token].add(row[5])
            labels.add(row[7].split(':')[0].lower())
    # Add gold labels and sentence IDs
    dataset = conllu_dataset.ConlluDataset()
    f = open('%s/%s-ud-%s.conllu' %(treebank_dir, tbid, test_type), 'rb')
    dataset.load_file(f)
    f.close()
    for sentence in dataset:
        sent_ids.add(sentence.sent_id)
        for row in sentence:
            labels.add(row[7].split(':')[0].lower())
    return vocab, morph, labels, sent_ids

def get_keys(labels, sent_ids):
    keys = set()
    for where in 'TRAM':
        for label in labels:
            keys.add('%s:%s' %(where, label))
        keys.add('%s:OOV' %where)
        keys.add('%s:MAW' %where)
        keys.add('%s:IV' %where)
        keys.add('%s:MUW' %where)
    for freq in ('0-0', '1-1', '2-2', '3-'):
        keys.add('S:OOV:%s' %freq)
        keys.add('S:MAW:%s' %freq)
        keys.add('S:IV:%s' %freq)
        keys.add('S:MUW:%s' %freq)
        for label in labels:
            keys.add('S:%s:%s' %(label, freq))
        keys.add('SL:12-12,S:OOV:%s' %freq)
        keys.add('SL:12-12,S:MAW:%s' %freq)
        keys.add('SL:12-12,S:IV:%s' %freq)
        keys.add('SL:12-12,S:MUW:%s' %freq)
        for label in labels:
            keys.add('SL:12-12,S:%s:%s' %(label, freq))
    for slrange in ('-9', '10-19', '12-12', '20-39', '40-'):
        keys.add('SL:%s' %slrange)
    keys.add('any')
    for sent_id in sent_ids:
        keys.add('ID:%s' %sent_id)
    return keys

def get_predictions_minrounds_and_last(workdirs, treebank, minrounds, test_type):
    global filename_extension
    minrounds = '%d' %minrounds
    short_lcode, lcode, language, tbid, tbname, long_tbname = treebank
    key2predictions = {}   # keys = {f, g, h} x {0, minrounds, last}
    augexp2lastround = {
        '0': 24,
        '2': 20,
        '4': 16,
        '6': 12,
        '8': 8,
        'A': 4,
    }
    for workdir in workdirs:
        for entry in os.listdir(workdir):
            if not entry.startswith(short_lcode):
                continue
            if len(entry) != 8:
                continue
            parser = entry[1]
            method = entry[2:7]
            augexp = entry[7]
            lastround = augexp2lastround[augexp]
            # scan folder for files like
            # prediction-12-E-ug_udt-dev-2LNqJcpnpadlgBWJ17nT.conllu
            predictions = []
            for filename in os.listdir('%s/%s' %(workdir, entry)):
                if not filename.startswith('prediction'):
                    continue
                if not filename.endswith(filename_extension):
                    continue
                fields = filename[:-len(filename_extension)].split('-')
                if fields[2] != 'E'        \
                or fields[4] != test_type  \
                or fields[3] != tbid       \
                or len(fields[5]) != 20:
                    continue
                tt_round = fields[1]
                while tt_round[0] == '0' and len(tt_round) > 1:
                    tt_round = tt_round[1:]
                tt_round = int(tt_round)
                if tt_round > lastround:
                    continue
                while tt_round >= len(predictions):
                    predictions.append(None)
                predictions[tt_round] = filename
            for tt_round in  ('0', minrounds, 'last'):
                key = '%s\t%s' %(parser, tt_round)
                if tt_round == 'last':
                    filename = predictions[-1]
                else:
                    filename = predictions[int(tt_round)]
                if filename is None:
                    raise ValueError('Missing prediction of round %s for parser %s with method %s, augexp %s and filenames %r' %(tt_round, parser, method, augexp, predictions))
                path = '%s/%s/%s' %(workdir, entry, filename)
                dataset = basic_dataset.load_or_map_from_filename(
                    conllu_dataset.new_empty_set(),
                    path,
                )
                if not key in key2predictions:
                    key2predictions[key] = {}
                key2 = '%s\t%s' %(method, augexp)
                if key2 in key2predictions[key]:
                    raise ValueError('duplicate prediction %r with key %r, 2nd key %r and treebank %r' %(filename, key, key2, treebank))
                key2predictions[key][key2] = dataset
    return key2predictions

def get_predictions_main_comparison(workdirs, treebank, test_type):
    short_lcode, lcode, language, tbid, tbname, long_tbname = treebank
    key2predictions = {}   # keys = {f, h} x {b, tt}
    for workdir in workdirs:
        for entry in os.listdir(workdir):
            match = False
            if entry.startswith('baseline-tt-sim') \
            and entry.endswith(filename_extension):
                # 0        1  2   3  4    5 6 7      8   9
                # baseline-tt-sim-en-udpf-E-0-en_ewt-dev-Al48bPGUC2ajR7wL9q0Q.conllu.bz2
                fields = entry.split('-')
                if len(fields) != 10 \
                or fields[5] != 'E' \
                or fields[7] != tbid \
                or fields[8] != test_type:
                    continue
                match = True
                parser = fields[4]
                tt_round = 'baseline'
            if entry.startswith('tt-') \
            and entry.endswith(filename_extension):
                # 0  1  2    3 4 5      6    7
                # tt-vi-udpf-E-6-vi_vtb-test-S24BZ3wApIfLJ7pccipY.conllu.bz2
                fields = entry.split('-')
                if len(fields) != 8 \
                or fields[3] != 'E' \
                or fields[5] != tbid \
                or fields[6] != test_type:
                    continue
                match = True
                parser = fields[2]
                tt_round = 'tritraining'
            if match:
                if parser == 'elmo':
                    parser = 'z_elmo'
                elif parser == 'udpf':
                    parser = 'a_udpf'
                key = '%s\t%s' %(parser, tt_round)
                path = '%s/%s' %(workdir, entry)
                dataset = basic_dataset.load_or_map_from_filename(
                    conllu_dataset.new_empty_set(),
                    path,
                )
                if not key in key2predictions:
                    key2predictions[key] = {}
                key2 = 'best12'
                if key2 in key2predictions[key]:
                    raise ValueError('duplicate prediction %r with key %r, 2nd key %r and treebank %r' %(path, key, key2, treebank))
                key2predictions[key][key2] = dataset
    return key2predictions

def is_oov(row, vocab):
    token = row[1]
    if not token in vocab:
        return True
    else:
        return False

def is_maw(row, token2morph):
    token = row[1]
    if token in token2morph:
        tr_morphs = token2morph[token]
        if len(tr_morphs) > 1:
            # token is ambiguous in the training data
            return True
        elif row[5] not in tr_morphs:
            # this test instance's gold morphology deviates
            # from what was observed in training
            return True
    return False

def has_label(row, label):
    if label == row[7].split(':')[0].lower():
        return True
    return False

def is_eval_row(row, sentence, rel_type, feature, vocab, token2morph):
    # always check the gold row
    if feature == 'OOV':
        if is_oov(row, vocab):
            return True
    elif feature == 'IV':
        if not is_oov(row, vocab):
            return True
    elif feature == 'MAW':
        if is_maw(row, token2morph):
            return True
    elif feature == 'MUW':
        if not is_maw(row, token2morph):
            return True
    else:
        if has_label(row, feature):
            return True
    if rel_type != 'T':
        # check the gold head
        head = int(row[6])
        if head > 0:
            # head 0 means it is the root, which has no head
            # within the sentence; we use 0 for the first token
            head_row = sentence[head-1]
            if feature == 'OOV':
                if is_oov(head_row, vocab):
                    return True
            elif feature == 'IV':
                if not is_oov(head_row, vocab):
                    return True
            elif feature == 'MAW':
                if is_maw(head_row, token2morph):
                    return True
            elif feature == 'MUW':
                if not is_maw(head_row, token2morph):
                    return True
            elif rel_type == 'R':
                if has_label(head_row, feature):
                    return True
    return False

def get_selections(predictions, treebank, vocab, token2morph, sent_ids, eval_keys, test_type):
    ''' for each test sentence and each evaluation key, determine
        which tokens are part of the evaluation
    '''
    selections = {}     # key = (eval_key, sentence_index), value = list of token indices
    _, _, _, tbid, _, long_tbname = treebank
    treebank_dir = get_treebank_dir(long_tbname)
    testset = conllu_dataset.ConlluDataset()
    f = open('%s/%s-ud-%s.conllu' %(treebank_dir, tbid, test_type), 'rb')
    testset.load_file(f)
    f.close()
    for sentence_index, sentence in enumerate(testset):
        sentence_length = len(sentence)
        for eval_key in eval_keys:
            eval_tokens = set(range(sentence_length))
            for constraint in eval_key.split(','):
                if constraint == 'any':
                    # equivalent to no constraint
                    pass
                elif constraint.startswith('ID:'):
                    # only evaluate if sentence ID matches
                    if constraint[3:] != sentence.sent_id:
                        eval_tokens = set()
                elif constraint.startswith('SL:'):
                    # only evaluate if length in given range
                    min_len, max_len = constraint[3:].split('-')
                    if min_len and sentence_length < int(min_len) :
                        eval_tokens = set()
                    if max_len and sentence_length > int(max_len) :
                        eval_tokens = set()
                elif constraint.startswith('S:'):
                    # only evaluate if sentence has given feature
                    # with given frequency
                    _, feature, freq_range = constraint.split(':')
                    count = 0
                    for row in sentence:
                        if feature == 'OOV':
                            if is_oov(row, vocab):
                                count += 1
                        elif feature == 'IV':
                            if not is_oov(row, vocab):
                                count += 1
                        elif feature == 'MAW':
                            if is_maw(row, token2morph):
                                count += 1
                        elif feature == 'MUW':
                            if not is_maw(row, token2morph):
                                count += 1
                        else:
                            if has_label(row, feature):
                                count += 1
                    # check that count is in given range
                    min_freq, max_freq = freq_range.split('-')
                    if min_freq and count < int(min_freq) :
                        eval_tokens = set()
                    if max_freq and count > int(max_freq) :
                        eval_tokens = set()
                elif constraint[:2] in ('T:', 'R:', 'A:', 'M:'):
                    rel_type, feature = constraint.split(':')
                    eval_tokens_2 = set()
                    for token_index, row in enumerate(sentence):
                        if is_eval_row(row, sentence, rel_type, feature, vocab, token2morph):
                            # always check gold for whether to evaluate this token
                            eval_tokens_2.add(token_index)
                        if rel_type in 'AM':
                            # check all predictions
                            found_instance = False
                            found_majority = False
                            for pkey1 in predictions:
                                partition = predictions[pkey1]
                                count = 0
                                total = len(partition)
                                needed_for_majority = int((total+2)/2)
                                for pkey2 in partition:
                                    pred_sentence = partition[pkey2][sentence_index]
                                    pred_row = pred_sentence[token_index]
                                    if is_eval_row(pred_row, pred_sentence, rel_type, feature, vocab, token2morph):
                                        found_instance = True
                                        count += 1
                                        if count >= needed_for_majority:
                                            found_majority = True
                                        if rel_type == 'A':
                                            break
                                        if rel_type == 'M' and found_majority:
                                            break
                                if rel_type == 'A' and found_instance:
                                    break
                                if rel_type == 'M' and found_majority:
                                    break
                            if rel_type == 'A' and found_instance:
                                eval_tokens_2.add(token_index)
                            if rel_type == 'M' and found_majority:
                                eval_tokens_2.add(token_index)
                    eval_tokens = eval_tokens.intersection(eval_tokens_2)
                else:
                    raise ValueError('unsupported token selector %s' %constraint)
            key = (eval_key, sentence_index)
            selections[key] = sorted(eval_tokens)
    return selections

def get_counts(predictions, selections, eval_keys, treebank, test_type):
    _, _, _, tbid, _, long_tbname = treebank
    treebank_dir = get_treebank_dir(long_tbname)
    testset = conllu_dataset.ConlluDataset()
    f = open('%s/%s-ud-%s.conllu' %(treebank_dir, tbid, test_type), 'rb')
    testset.load_file(f)
    f.close()
    key2counts = {}
    for sentence_index, gold_sentence in enumerate(testset):
        sentence_length = len(gold_sentence)
        for eval_key in eval_keys:
            eval_tokens = selections[(eval_key, sentence_index)]
            for pkey1 in predictions:
                partition = predictions[pkey1]
                try:
                    correct, total = key2counts[(eval_key, pkey1)]
                except KeyError:
                    correct, total = 0, 0
                for pkey2 in partition:
                    prediction = partition[pkey2][sentence_index]
                    if len(prediction) != sentence_length:
                        raise ValueError('Wrong sentence length of prediction %r, %r at sentence index %d with evaluation key %r' %(pkey1, pkey2, sentence_index, eval_key))
                    total += len(eval_tokens)
                    for token_index in eval_tokens:
                        gold_row = gold_sentence[token_index]
                        pred_row = prediction[token_index]
                        gold_head = gold_row[6]
                        pred_head = pred_row[6]
                        gold_label = gold_row[7]
                        pred_label = pred_row[7]
                        # head and label correct?
                        if gold_head == pred_head and gold_label == pred_label:
                            correct += 1
                key2counts[(eval_key, pkey1)] = correct, total
    return key2counts

def write_tsv(
    f, counts, pkey_part, pkey_name, other_name, eval_keys,
    treebank, test_type
):
    if not pkey_part in (0,1):
        raise NotImplementedError
    other_keyparts = set()
    target_keyparts = set()
    for eval_key, pkey in counts:
        parts = pkey.split()
        other_keyparts.add(parts[1-pkey_part])
        target_keyparts.add(parts[pkey_part])
    other_keyparts = sorted(other_keyparts)
    target_keyparts = sorted(target_keyparts)
    num_other_keyparts = len(other_keyparts)
    num_target_keyparts = len(target_keyparts)
    comparisons = []
    # header
    header = []
    header.append(other_name)
    header.append('constraint')
    header.append('events')
    for target in target_keyparts:
            header.append(target)
    for i in range(num_target_keyparts - 1):
        for j in range(i+1, num_target_keyparts):
            comparisons.append((target_keyparts[i], target_keyparts[j]))
            header.append('%s->%s' %(comparisons[-1]))
    has_sent_id = False
    for key in eval_keys:
        if key.startswith('ID:'):
            has_sent_id = True
            break
    if has_sent_id:
        header.append('Text')
        _, _, _, tbid, _, long_tbname = treebank
        treebank_dir = get_treebank_dir(long_tbname)
        testset = conllu_dataset.ConlluDataset()
        f_testset = open('%s/%s-ud-%s.conllu' %(treebank_dir, tbid, test_type), 'rb')
        testset.load_file(f_testset)
        f_testset.close()
    f.write('\t'.join(header))
    f.write('\n')
    not10x = False
    for other_keypart in other_keyparts:
        for eval_key in sorted(eval_keys):
            totals = []
            for target_keypart in target_keyparts:
                if pkey_part == 0:
                    pkey1 = '%s\t%s' %(target_keypart, other_keypart)
                else:
                    pkey1 = '%s\t%s' %(other_keypart, target_keypart)
                totals.append(counts[(eval_key, pkey1)][1])
            totals.sort()
            if not totals or totals[0] != totals[-1]:
                raise ValueError('inconsistent totals %r for %r' %(totals, (eval_key, pkey1)))
            row = []
            row.append(other_keypart)
            row.append(eval_key)
            #row.append('%.1f' %(totals[0]/10.0))
            row.append('%.0f' %(totals[0]))
            if totals[0] % 10:
                not10x = True
            for target in target_keyparts:
                if pkey_part == 0:
                    pkey1 = '%s\t%s' %(target, other_keypart)
                else:
                    pkey1 = '%s\t%s' %(other_keypart, target)
                correct, total = counts[(eval_key, pkey1)]
                if total:
                    score = 100.0 * correct / float(total)
                    row.append('%.9f' %score)
                else:
                    row.append('N/A')
            for target_1, target_2 in comparisons:
                if pkey_part == 0:
                    pkey1 = '%s\t%s' %(target_1, other_keypart)
                else:
                    pkey1 = '%s\t%s' %(other_keypart, target_1)
                correct, total = counts[(eval_key, pkey1)]
                if total:
                    score_1 = 100.0 * correct / float(total)
                    if pkey_part == 0:
                        pkey1 = '%s\t%s' %(target_2, other_keypart)
                    else:
                        pkey1 = '%s\t%s' %(other_keypart, target_2)
                    correct, total = counts[(eval_key, pkey1)]
                    if total:
                        score_2 = 100.0 * correct / float(total)
                        row.append('%.9f' %(score_2-score_1))
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            if has_sent_id and eval_key.startswith('ID:'):
                sent_id = eval_key[3:]
                text = []
                for sentence in testset:
                    if sentence.sent_id == sent_id:
                        for sent_row in sentence:
                            text.append(sent_row[1])
                        break
                text = ' '.join(text)
                if '"' in text:
                    # https://webapps.stackexchange.com/questions/26841/how-to-escape-a-double-quote-in-a-tsv-file-for-import-into-a-google-sheets
                    # (for description in English: 2nd comment of accepted
                    # answer; no source code copied)
                    text = '"%s"' %(text.replace('"', '""'))
                row.append(text)
            f.write('\t'.join(row))
            f.write('\n')
    if not10x:
        sys.stderr.write('At least one total of events was not a multiple of 10.\n')

def get_eval_keys_subsets(eval_keys):
    subsets = {}
    for key in eval_keys:
        if key == 'any':
            continue
        if key.startswith('ID:'):
            skey = 'by-sentence'
        elif key.startswith('T:'):
            skey = 'token-rows'
        elif key.startswith('R:'):
            skey = 'gold-relation'
        elif key.startswith('A:'):
            skey = 'any-prediction'
        elif key.startswith('M:'):
            skey = 'majority-predictions'
        elif key.startswith('S:'):
            skey = 'sentence-level'
        elif key.startswith('SL:12-12'):
            skey = 'length-12'
        elif key.startswith('SL:'):
            skey = 'by-length'
        else:
            skey = 'other'
        if skey not in subsets:
            subsets[skey] = set()
        subsets[skey].add(key)
    for skey in subsets:
        subsets[skey].add('any')
    return subsets.items()

class FileUpdater():

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        if os.path.exists(self.path):
            self.f = open(self.path, 'rb')
            self.m = 'check'
        else:
            self.f = open(self.path, 'wb')
            self.m = 'write'
        return self

    def __exit__(self, type, value, traceback):
        if self.m == 'check':
            pos = self.f.tell()
            if self.f.read(1):
                self.f.seek(0)
                keep = self.f.read(pos)
                self.f.close()
                self.f = open(self.path, 'wb')
                self.f.write(keep)
        self.f.close()

    def write(self, data):
        if self.m == 'check':
            pos = self.f.tell()
            if self.f.read(len(data)) == data:
                return len(data)
            else:
                self.f.seek(0)
                keep = self.f.read(pos)
                self.f.close()
                self.f = open(self.path, 'wb')
                self.f.write(keep)
                self.m = 'write'
                return self.f.write(data)
        else:
            return self.f.write(data)

def main():
    opt_mode = 'main-comparison'
    workdirs = sys.argv[1:]
    test_type = 'dev'
    treebanks, minrounds = get_treebanks_and_minrounds(workdirs)
    for treebank in treebanks:
        vocab, token2morph, labels, sent_ids = get_vocab_and_morph_and_labels_and_sent_ids(treebank, test_type)
        keys = get_keys(labels, sent_ids)
        with FileUpdater('%s/ea-evalkeys-%s.txt' %(treebank[2], treebank[1])) as f:
            for key in sorted(keys):
                f.write('%s\n' %key)
        if opt_mode == 'minrounds-and-last-round':
            predictions = get_predictions_minrounds_and_last(
                workdirs, treebank, minrounds, test_type
            )
        elif opt_mode == 'main-comparison':
            predictions = get_predictions_main_comparison(
                workdirs, treebank, test_type
            )
        else:
            raise ValueError('unknown mode %s' %opt_mode)
        with FileUpdater('%s/ea-predictions-%s.txt' %(treebank[2], treebank[1])) as f:
            for key in sorted(predictions.keys()):
                for key2 in sorted(predictions[key].keys()):
                    f.write('%s\t%s\t%s\n' %(key, key2, predictions[key][key2].filename))
        selections = get_selections(predictions, treebank, vocab, token2morph, sent_ids, keys, test_type)
        counts = get_counts(predictions, selections, keys, treebank, test_type)
        for pkey_part, pkey_name, other_name in [
            (0, 'parser', 'round'),
            (1, 'round',  'parser'),
        ]:
            for eval_subset_name, eval_keys_subset in get_eval_keys_subsets(keys):
                with FileUpdater('%s/ea-effect-of-%s-for-%s-%s.tsv' %(
                    treebank[2], pkey_name, treebank[1], eval_subset_name
                )) as f:
                    write_tsv(
                        f, counts, pkey_part, pkey_name, other_name,
                        eval_keys_subset,
                        treebank, test_type,
                    )

if __name__ == "__main__":
    main()

