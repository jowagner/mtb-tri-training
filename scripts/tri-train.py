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

import importlib
import hashlib
import os
import random
import string
import sys

import basic_dataset

def print_usage():
    print('Usage: %s [options]' %(os.path.split(sys.argv[0])[-1]))
    print("""
Options:

    --test-type  STRING     whether to test on the dev or test section
                            (default: dev)

    --workdir  DIR          Path to working directory
                            (default: . = current directory)

    --labelled  STRING
    --unlabelled  STRING    append data sets to the list of labelled or
                            unlabelled data sets; STRING must be a space- or
                            colon-separated list of data set IDs;
                            the module specified with --dataset-module
                            interprets these IDs to load the data

    --init-seed  STRING     initialise random number generator with STRING;
                            (default or empty string: use system seed)

    --model-init  TYPE      derive the initialisation seed for each model to
                            be trained from
                            the output of this script's random number
                            generator; types:
                                int15 = 15 bit non-zero number,
                                int31 = 31 bit non-zero number,
                                int63 = 63 bit non-zero number,
                                str12 = 12 characters (1 letter and 11 letters
                                        or digits)
                                str24 = 2x str12
                                compose = concatenation of our seed (--seed),
                                    learner rank (1-3) and training round,
                                    i.e. adding unique additional digits to the
                                    main seed
                            (default: do not provide the models with a seed,
                            usually meaning that they use a system seed)

    --dataset-module  NAME  handle data sets using the module NAME;
                            combine with --help to see module-specific options
                            (default: conllu_dataset)

    --model-module  NAME    train models using the module NAME;
                            combine with --help to see module-specific options;
                            specify 3 times to mix different models in
                            tri-training
                            (default: uuparser_model)

    --seed-size  NUMBER     How many tokens to sample (with replacement) from
                            the labelled data for each learner.
                            As full sentences are selected the actual number
                            of selected tokens can deviate from the requested
                            number.
                            If NUMBER end with % it is relative to the size
                            of the labelled data.
                            (Default: 100.0%)

    --seed-attempts  NUMBER  Create NUMBER samples and pick, for each learner,
                            the one that is closest to the desired seed size.
                            (Default: 5)

    --subset-size  NUMBER   How many items to select from the unlabelled
                            data in each tri-training iteration
                            As full sentences are selected the actual number
                            of selected tokens can deviate from the requested
                            number.
                            If NUMBER end with % it is relative to the size
                            of the labelled data.
                            (Default: 20k)

    --subset-attempts  NUMBER  Create NUMBER subsets and pick the one that is
                            closest to the desired subset size
                            (default: 5)

    --augment-size  NUMBER  How many items to add to each learner in each
                            tri-training iteration.
                            If the subset size is too small, e.g. less than 3x
                            augment size, the augment size may not be reached.
                            As full sentences are selected the actual number
                            of selected tokens can deviate from the requested
                            number.
                            (default: 4k)

    --augment-attempts  NUMBER  Create NUMBER augmentation sets and pick the
                            one that is closest to the desired set size
                            (default: 5)

    --diversify-attempts  NUMBER  Iteratively grow the subset picking the
                            sentence which the largest distance to the
                            sentences selected so far among NUMBER randomly
                            picked sentences.
                            The same procedure is applied to the augmentation
                            set if the candidate augmentation set exceeds the
                            augment size.
                            (Default: 1 = pick subset at random)

    --oversample            In each iteration and for each learner, oversample
                            seed data to match size of unlabelled data.
                            (Default: use seed data as is)

    --iterations  NUMBER    Perform NUMBER iterations of tri-training.
                            (Default: 5)

    --learners  NUMBER      Use NUMBER learners in tri-training. Knowledge
                            transfer is always from 2 teachers to 1 learner.
                            (Default: 3)

    --last-k  NUMBER        Only use the automatically labelled data of the
                            last k tri-training iterations
                            (default: 0 = use all iterations)

    --decay  NUMBER         Subsample (without replacement) fraction
                            NUMBER^(j-i) of automatically labelled data of the
                            i-th iteration in iteration j;
                            can be combined with --last-k
                            (default: 1.0 = use all data)

    --iteration-selection   Use development data to select strongest model for
                            each learner.
                            (Default: Use model of the last tri-training
                            interation for each learner.)

    --epoch-selection  MODE  How to select the epoch for each model:
                            dev  = use development data that is part of the
                                   data set
                            last = use last epoch
                            remaining = use labelled data not part of the
                                   seed data (due to sampling with
                                   replacement)
                            9010 = split seed data 90:10 into train and dev

    --per-item              Apply apply tri-training to individual items,
                            either select an item or not.
                            (Default: only select full sentences)

    --per-item-and-target   Apply apply tri-training to individual target
                            features of items, e.g. in depdendency parsing
                            make independent decisions for heads and labels.

    --continue              Skip steps finished in a previous run
                            (default: abort if intermediate output files are
                            found)


""")

def main():
    opt_help  = False
    test_type = 'dev'
    opt_workdir = '.'
    opt_debug = False
    opt_init_seed  = None
    opt_labelled_ids = []
    opt_unlabelled_ids = []
    opt_dataset_module = 'conllu_dataset'
    opt_model_modules  = []
    opt_model_init_type = None
    opt_seed_size = '100.0%'
    opt_seed_attempts = 5
    opt_subset_size = '20k'
    opt_subset_attempts = 5
    opt_augment_size = '4k'
    opt_augment_attempts = 5
    opt_diversify_attempts = 1
    opt_oversample = False
    opt_iterations = 5
    opt_learners = 3
    opt_last_k = 0
    opt_last_decay = 1.0

    while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
        option = sys.argv[1]
        del sys.argv[1]
        if option in ('--help', '-h'):
            opt_help = True
            break
        elif option == '--test-type':
            test_type = sys.argv[1]
            del sys.argv[1]
        elif option == '--workdir':
            opt_workdir = sys.argv[1]
            del sys.argv[1]
        elif option == '--init-seed':
            opt_init_seed = sys.argv[1]
            del sys.argv[1]
        elif option in ('--model-init', '--model-init-type'):
            opt_model_init_type = sys.argv[1]
            del sys.argv[1]
        elif option in ('--labelled', '--unlabelled'):
            for tbid in sys.argv[1].replace(':', ' ').split():
                if option == '--labelled':
                    opt_labelled_ids.append(tbid)
                else:
                    opt_unlabelled_ids.append(tbid)
            del sys.argv[1]
        elif option == '--dataset-module':
            opt_dataset_module = sys.argv[1]
            del sys.argv[1]
        elif option == '--model-module':
            opt_model_modules.append(sys.argv[1])
            del sys.argv[1]
        elif option == '--seed-size':
            opt_seed_size = sys.argv[1]
            del sys.argv[1]
        elif option == '--seed-attempts':
            opt_seed_attempts = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--subset-size':
            opt_subset_size = sys.argv[1]
            del sys.argv[1]
        elif option == '--subset-attempts':
            opt_subset_attempts = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--augment-size':
            opt_augment_size = sys.argv[1]
            del sys.argv[1]
        elif option == '--augment-attempts':
            opt_augment_attempts = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--diversify-attempts':
            opt_diversify_attempts = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--oversample':
            opt_oversample = True
        elif option == '--iterations':
            opt_iterations = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--learners':
            opt_learners = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--last-k':
            opt_last_k = int(sys.argv[1])
            del sys.argv[1]
        elif option in ('--last-decay', '--decay'):
            opt_last_decay = float(sys.argv[1])
            del sys.argv[1]
        elif option == '--debug':
            opt_debug = True
        else:
            print('Unsupported or not yet implemented option %s' %option)
            opt_help = True
            break

    if len(sys.argv) != 1:
        opt_help = True

    if opt_help:
        print_usage()
        sys.exit(0)

    if opt_learners < 1:
        raise ValueError('must have at least one learner')
    if opt_last_k < 0:
        raise ValueError('last_k must not be negative')
    if opt_last_decay > 1.0:
        raise ValueError('last_decay must not be greater than 1')
    if opt_last_decay <= 0.0:
        raise ValueError('last_decay must not be zero or negative')

    if not opt_model_modules:
        opt_model_modules.append('uuparser_model')

    dataset_module = importlib.import_module(opt_dataset_module)

    training_data_sets = []
    dev_sets = []
    test_sets = []
    for dataset_id in opt_labelled_ids:
        tr, dev, test = dataset_module.load(dataset_id)
        #print('Dataset %r: %r, %r, %r' %(dataset_id, tr, dev, test))
        training_data_sets.append(tr)
        dev_sets.append(dev)
        test_sets.append(test)
    training_data = basic_dataset.Concat(training_data_sets)

    unlabelled_data_sets = []
    for dataset_id in opt_unlabelled_ids:
        tr, _, _ = dataset_module.load(
            dataset_id, load_dev = False, load_test = False
        )
        #print('Dataset %r: %r' %(dataset_id, tr))
        unlabelled_data_sets.append(tr)
    unlabelled_data = basic_dataset.Concat(unlabelled_data_sets)

    training_data_size = training_data.get_number_of_items()
    unlabelled_data_size = unlabelled_data.get_number_of_items()
    opt_seed_size    = adjust_size(opt_seed_size,    training_data_size)
    opt_subset_size  = adjust_size(opt_subset_size,  unlabelled_data_size)
    opt_augment_size = adjust_size(opt_augment_size, unlabelled_data_size)

    tr_size = training_data.get_number_of_items()
    print('labelled training data with %d items in %d sentences' %(
        tr_size, len(training_data)
    ))
    print('opt_seed_size', opt_seed_size)
    print('labelled training data with %d items in %d sentences' %(
        unlabelled_data.get_number_of_items(),
        len(unlabelled_data)
    ))
    print('opt_subset_size', opt_subset_size)
    print('opt_augment_size', opt_augment_size)

    print('\n== Selection of Seed Data ==\n')

    if opt_init_seed:
        random.seed(int(hashlib.sha512('seed selection %s' %(
            opt_init_seed,
        )).hexdigest(), 16))

    seed_sets = []
    for learner_rank in range(opt_learners):
        candidates = []
        for _ in range(opt_seed_attempts):
            n_sentences = int(0.5 + len(training_data) * opt_seed_size / tr_size)
            candidate = basic_dataset.Sample(training_data, random, n_sentences)
            size = candidate.get_number_of_items()
            deviation = abs(size - opt_seed_size)
            candidates.append((deviation, random.random(), candidate))
        candidates.sort()
        best_deviation, _, best_seed_set = candidates[0]
        print('Learner %d has seed data with %d items in %d sentences' %(
            learner_rank+1, best_seed_set.get_number_of_items(), len(best_seed_set),
        ))
        tr_data_filename = '%s/seed-set-%d.conllu' %(opt_workdir, learner_rank+1)
        f_out = open(tr_data_filename, 'w')
        best_seed_set.save_to_file(f_out)
        f_out.close()
        seed_sets.append(best_seed_set)

    print('\n== Training of Seed Models ==\n')

    for learner_rank in range(opt_learners):
        print('Learner:', learner_rank+1)
        model_init_seed = get_model_seed(opt_model_init_type, opt_init_seed, learner_rank, 0)
        print('Model initialisation seed:', model_init_seed)
        data_fingerprint = seed_sets[learner_rank].hexdigest()
        print('Seed data fingerprint:', data_fingerprint)

        # TODO: check whether model is ready and train it if not

    for training_round in range(opt_iterations):
        print('\n== Tri-training Iteration %d of %d ==\n' %(
            training_round+1, opt_iterations
        ))
        if opt_init_seed:
            random.seed(int(hashlib.sha512('round %d: %s' %(
                training_round, opt_init_seed,
            )).hexdigest(), 16))
        print('Selecting subset of unlabelled data:')
        picked = {}

        print('Making predictions:')

        # TODO: call parser

        print('Teaching:')

        new_datasets = []
        for _ in range(opt_learners):
            new_datasets.append(dataset_module.new_empty_set())
        for s_index, s_predictions in enumerate(dataset_predictions):
            learner_index, merged_prediction = knowledge_transfer(
                s_predictions, column_weights, opt_learners,
                opt_max_teacher_disagreement_fraction,
                opt_min_teacher_agreements,
                opt_learner_must_disagree,
                opt_min_learner_disagreement,
            )
            if learner_index < 0:
                continue
            # TODO: The above constraints are sentence-level
            #       filters. We also need to apply similar
            #       contraints per item and column, i.e.
            #       optionally only accept items from the
            #       merged prediction that sufficiently
            #       disagree from the learner's prediction
            #       (or are undefined, i.e. '_').

            # add new sentence to data set of learner
            new_datasets[learner_index].append(merged_prediction)

        for learner_index in range(opt_learners):
            # write new labelled data to file
            # TODO
            tr_data_filename = '%s/new-set-%02d-%d.conllu' %(opt_workdir, training_round+1, learner_rank+1)
            f_out = open(tr_data_filename, 'w')
            new_datasets[learner_index].save_to_file(f_out)
            f_out.close()


        print('Training of new models:')
        for learner_rank in range(opt_learners):
            print(' * Learner %d, seed %r' %(learner_rank+1,
                get_model_seed(opt_model_init_type, opt_init_seed, learner_rank, training_round+1)))

    print('\n== Final Model ==\n')
    # TODO

def get_disagreement(prediction1, prediction2, column_weights):
    raise NotImplementedError

def merge_predictions(predictions):
    raise NotImplementedError

def knowledge_transfer(
    predictions, column_weights, learners = 3,
    max_teacher_disagreement_fraction = 0.20,
    min_teacher_agreements = 3.25,
    learner_must_disagree = True, min_learner_disagreement = 1.25
):
    # knowledge transfer candidates: first element says how much the teachers disagree
    kt_candidates = []
    s_length = float(len(predictions[0]))
    total = 0.0
    for weight in column_weights:
        total += weight * s_length
    for learner_index in range(learners):
        learner_prediction = predictions[learner_index]
        if learners == 1:
            t1_indices = [None]
        else:
            t1_indices = range(learners-1)
        for teacher1_index in t1_indices:
            if learners > 1:
                t1_prediction = predictions[teacher1_index]
                if teacher1_index == learner_index:
                    continue
            if learners <= 2:
                t2_indices = [None]
            else:
                t2_indices = range(teacher1_index+1, learners)
            for teacher2_index in t2_indices:
                if learners == 1:
                    # self-training
                    teachers_predictions = [learner_prediction]
                elif learners == 2:
                    # co-training
                    teachers_predictions = [t1_prediction]
                else:
                    if teacher2_index == learner_index:
                        continue
                    t2_prediction = predictions[teacher2_index]
                    # measure disagreement between teachers
                    teacher_disagreement = get_disagreement(
                        t1_prediction, t2_prediction, column_weights
                    )
                    teacher_disagreement_fraction = teacher_disagreement / s_length
                    if teacher_disagreement_fraction > max_teacher_disagreement_fraction:
                        continue
                    if total - teacher_disagreement < min_teacher_agreements:
                        continue
                    teachers_predictions = [
                        t1_prediction, t2_prediction
                    ]
                # optionally consider disagreement with learner
                if learner_must_disagree:
                    merged_prediction = merge_predictions(teachers_predictions)
                    learner_disagreement = get_disagreement(
                        learner_prediction, merged_prediction, column_weights
                    )
                    if learner_disagreement < min_learner_disagreement:
                        # skip this candidate as the learner
                        # is unlikely to learn much from it
                        continue
                else:
                    # postpone merging to when it is needed
                    merged_prediction = None
                    learner_disagreement = 0
                # record candidate
                priority = (teacher_disagreement, -learner_disagreement)
                kt_candidates((
                    priority, random.random(),
                    learner_index, teacher1_index, teacher2_index,
                    merged_prediction
                ))
    if not kt_candidates:
        # can happen when predictions agree and learner_must_disagree is set
        return -1, None
    kt_candidates.sort()
    _, _, learner_index, t1_index, t2_index, prediction = kt_candidates[0]
    if prediction is None:
        t1_prediction = predictions[t1_index]
        t2_prediction = predictions[t2_index]
        prediction = merge_predictions([
            t1_prediction, t2_prediction
        ])
    return learner_index, prediction

def adjust_size(size, data_size):
    if size.endswith('%'):
        fraction = float(size[:-1]) / 100.0
        return int(0.5 + fraction * data_size)
    elif size[-1:].lower() in 'kmgt':
        fraction = float(size[:-1])
        multiplier = {
            'k': 1000.0,
            'm': 10.0**6,
            'g': 10.0**9,
            't': 10.0**12,
        }[size[-1:].lower()]
        return int(0.5 + fraction * multiplier)
    else:
        return int(size)

def get_model_seed(mode, main_seed, learner_rank, training_round):
    if mode in ('system', None):
        return None
    elif mode.startswith('int'):
        bits = int(mode[3:])
        if bits < 1:
            raise ValueError('model seed must have at least 1 bit')
        return '%d' %random.randrange(1, 2**bits)
    elif mode.startswith('str'):
        n = int(mode[3:])
        if n < 1:
            raise ValueError('model seed must have at least 1 character')
        chars = []
        chars.append(random.choice(string.ascii_letters))
        n -= 1
        while n > 0:
            chars.append(random.choice(string.ascii_letters + string.digits))
            n -= 1
        return ''.join(chars)
    elif mode in ('combine', 'compose'):
        if main_seed is None:
            raise ValueError('cannot combine model seed without main seed')
        return '%s%d%02d' %(main_seed, learner_rank, training_round)
    else:
        raise ValueError('unknown model seed type %r' %mode)


if __name__ == "__main__":
    main()

