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

    --final-test            Also report results for the test sets of each
                            data set.
                            (Default: only report development set results)

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

    --manually-train        Quit this script each time models need to be
                            trained.
                            Tri-training can be continued after manually
                            creating the model files by re-running this
                            script with the same initialisation seed and
                            using the option --continue.
                            (Default: Ask the model module to train each
                            model.)

    --manually-predict      Quit this script each time predictions need to
                            be made, i.e. before running the predictor.
                            Tri-training can be continued after manually
                            creating the prediction file(s) by re-running
                            this script with the same initialisation seed
                            and using the option --continue.
                            (Default: Ask the model module to make
                            predictions.)

    --quit-after-prediction  Quit this script after all predictions for a
                            tri-training iteration have been made. After
                            inspecting and/or modifying the predictions,
                            tri-training can be continued by re-running this
                            script with the same initialisation seed and
                            using the option --continue.
                            (Default: Proceed to knowledge transfer between
                            learners without further ado.)

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
                            (default: udpipe_future)

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
                            (Default: 600k)

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
                            (default: 10k)

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

    --last-decay  NUMBER    Subsample (without replacement) fraction
                            NUMBER^(j-i) of automatically labelled data of the
                            i-th iteration in iteration j;
                            can be combined with --last-k
                            (default: 1.0 = use all data)

    --last-decay-attempts  NUMBER
                            Create NUMBER decayed sets and pick the
                            one that is closest to the desired set size
                            (default: 5)

    --epoch-selection  MODE
    --iteration-selection  MODE
                            How to select the epoch for each model and how to
                            select the tri-training iteration for the final
                            model:
                            dev  = use development data that is part of the
                                   data set (concatenation if multiple sets)
                            last = use last epoch
                            remaining = use labelled data not part of the
                                   seed data (due to sampling with
                                   replacement)
                            dev+remaining = concatenation of dev and remaining
                            9010 = split seed data 90:10 into train and dev
                            (Default: Select epoch using dev+remaining and
                            use last tri-training iteration for final models.)

    --max-selection-size  NUMBER
                            If validation set for model selection exceeds
                            NUMBER items subsample (without replacement)
                            the set to approximately NUMBER items.
                            Applied to both epoch and iteration selection.
                            (Default: 50k)

    --selection-attempts  NUMBER
                            How hard to try to reach the requested selection
                            size.
                            (Default: 5)

    --mask-teacher-disagreement  NUMBER
                            When the 2 teachers disagree about a label if an
                            item, replace it with '_' (which finally is
                            replaced with a random label) with probability
                            NUMBER. Otherwise pick the prediction of one of
                            the teachers at random.
                            (Default: 1.0 = always replace with '_')

    --mask-learner-agreement  NUMBER
                            When the learner agrees with the teachers'
                            joint prediction, reaplace it with '_' (which
                            finally is replaced with a random label) with
                            probability NUMBER. Otherwise reinforce
                            existing knowledge.
                            (Default: 0.0 = never replace with '_')

    # TODO: Not clear how the following two options apply. Maybe they should
    #       be short-hands for settings various other options (some of which
    #       are not yet documented here).

    --per-item              Apply apply tri-training to individual items,
                            either select an item or not.
                            (Default: only select full sentences)

    --per-item-and-target   Apply apply tri-training to individual target
                            features of items, e.g. in depdendency parsing
                            make independent decisions for heads and labels.

    --verbose               More detailed log output

    --continue              Skip steps finished in a previous run
                            (default: abort if intermediate output files are
                            found)


""")

def main():
    opt_help  = False
    opt_verbose = False
    opt_final_test = False
    opt_workdir = '.'
    opt_debug = False
    opt_init_seed  = None
    opt_manually_train = False
    opt_manually_predict = False
    opt_quit_after_prediction = False
    opt_labelled_ids = []
    opt_unlabelled_ids = []
    opt_dataset_module = 'conllu_dataset'
    opt_model_modules  = []
    opt_model_init_type = None
    opt_seed_size = '100.0%'
    opt_seed_attempts = 5
    opt_subset_size = '600k'
    opt_subset_attempts = 5
    opt_augment_size = '10k'
    opt_augment_attempts = 5
    opt_diversify_attempts = 1
    opt_oversample = False
    opt_iterations = 5
    opt_learners = 3
    opt_last_k = 0
    opt_last_decay = 1.0
    opt_last_decay_attempts = 5
    opt_epoch_selection = 'dev+remaining'
    opt_iteration_selection = 'last'
    opt_max_selection_size = '50k'
    opt_selection_attempts = 5
    opt_continue = False

    while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
        option = sys.argv[1]
        option = option.replace('_', '-')
        del sys.argv[1]
        if option in ('--help', '-h'):
            opt_help = True
            break
        elif option == '--final-test':
            opt_final_test = True
            del sys.argv[1]
        elif option == '--workdir':
            opt_workdir = sys.argv[1]
            del sys.argv[1]
        elif option == '--init-seed':
            opt_init_seed = sys.argv[1]
            del sys.argv[1]
        elif option == '--manually-train':
            opt_manually_train = True
        elif option == '--manually-predict':
            opt_manually_predict = True
        elif option == '--quit-after-prediction':
            opt_quit_after_prediction = True
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
        elif option == '--last-decay-attempts':
            opt_last_decay_attempts = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--epoch-selection':
            opt_epoch_selection = sys.argv[1]
            del sys.argv[1]
        elif option == '--iteration-selection':
            opt_iteration_selection = sys.argv[1]
            del sys.argv[1]
        elif option == '--max-selection-size':
            opt_max_selection_size = sys.argv[1]
            del sys.argv[1]
        elif option == '--selection-attempts':
            opt_selection_attempts = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--continue':
            opt_continue = True
        elif option == '--verbose':
            opt_verbose = True
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
        opt_model_modules.append('udpipe_future')

    dataset_module = importlib.import_module(opt_dataset_module)
    target_columns = dataset_module.get_target_columns()
    filename_extension = dataset_module.get_filename_extension()

    training_data_sets = []
    dev_sets = []
    test_sets = []
    for dataset_id in opt_labelled_ids:
        tr, dev, test = dataset_module.load(
            dataset_id, load_test = opt_final_test
        )
        #print('Dataset %r: %r, %r, %r' %(dataset_id, tr, dev, test))
        training_data_sets.append(tr)
        dev_sets.append(dev)
        test_sets.append(test)
    training_data = basic_dataset.Concat(training_data_sets)

    target_labelsets = []
    for column in target_columns:
        labelset = {}
        training_data.collect_labels(labelset, column)
        labelset = list(labelset.keys())
        labelset.sort()
        target_labelsets.append(labelset)

    unlabelled_data_sets = []
    unl_dev_sets = []
    unl_test_sets = []
    for dataset_id in opt_unlabelled_ids:
        tr, dev, test = dataset_module.load(
            dataset_id, load_test = opt_final_test
        )
        #print('Dataset %r: %r' %(dataset_id, tr))
        unlabelled_data_sets.append(tr)
        unl_dev_sets.append(dev)
        unl_test_sets.append(test)
    unlabelled_data = basic_dataset.Concat(unlabelled_data_sets)

    monitoring_datasets = []
    for dataset in dev_sets + test_sets + unl_dev_sets + unl_test_sets:
        if dataset is not None:
            monitoring_datasets.append(dataset)

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
    print('unlabelled training data with %d items in %d sentences' %(
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
    epoch_selection_sets = []
    iteration_selection_sets = []
    for learner_index in range(opt_learners):
        learner_rank = learner_index + 1
        seed_set = get_subset(
            training_data, opt_seed_size, random, opt_seed_attempts,
            with_replacement = True
        )
        print('Learner %d has seed data with %d items in %d sentences' %(
            learner_rank, seed_set.get_number_of_items(), len(seed_set),
        ))
        write_dataset(
            seed_set,
            '%s/seed-set-%d.conllu' %(opt_workdir, learner_rank)
        )
        # before we can use the seed set, we may have to slice off 10%
        if opt_epoch_selection == '9010' or opt_iteration_selection == '9010':
            seed_set_90 = get_subset(
                seed_set, int(0.5+0.90*opt_seed_size), random, opt_seed_attempts,
                with_replacement = False,
                write_file = \
                '%s/seed-subset-90-%d.conllu' %(opt_workdir, learner_rank)
            )
            seed_set_10 = get_remaining(
                seed_set_90, random,
                write_file = \
                '%s/seed-subset-10-%d.conllu' %(opt_workdir, learner_rank)
            )
            seed_sets.append(seed_set_90)
        else:
            seed_set_90 = None
            seed_set_10 = None
            seed_sets.append(seed_set)
        # create datasets for epoch and iteration selection
        for selection_sets, selection_type, name in [
            (epoch_selection_sets,     opt_epoch_selection,     'epoch'),
            (iteration_selection_sets, opt_iteration_selection, 'iteration'),
        ]:
            selection_sets.append(get_model_selection_dataset(
                selection_type, dev_sets, seed_set, seed_set_10,
                opt_max_selection_size, opt_selection_attempts, random,
                write_file = \
                '%s/for-%s-selection-%d.conllu' %(opt_workdir, name, learner_rank)
            ))

    print('\n== Training of Seed Models ==\n')

    model_modules = []
    if not opt_manually_train:
        for name in opt_model_modules:
            model_modules.append(importlib.import_module(name))

    models = train_models(
        opt_learners, seed_sets, epoch_selection_sets, model_modules,
        opt_model_init_type, opt_init_seed, 0,
        opt_workdir, opt_manually_train, opt_continue,
        opt_verbose,
        monitoring_datasets = monitoring_datasets
    )

    # evaluate models using all dev sets (and test sets if --final-test)
    evaluate(
        models,
        dev_sets + unl_dev_sets,
        test_sets + unl_test_sets,
        opt_labelled_ids + opt_unlabelled_ids,
        filename_extension = filename_extension,
        opt_workdir = opt_workdir,
        opt_continue = opt_continue,
        opt_verbose = opt_verbose
    )

    drop_all_targets = basic_dataset.SentenceDropout(
        rng = random.Random(0),
        target_columns = target_columns,
        dropout_probabilities = len(target_columns) * [1.0]
    )
    previously_picked = {}
    new_datasets = []
    for training_index in range(opt_iterations):
        training_round = training_index + 1
        print('\n== Tri-training Iteration %d of %d ==\n' %(
            training_round, opt_iterations
        ))
        if opt_init_seed:
            random.seed(int(hashlib.sha512('round %d: %s' %(
                training_round, opt_init_seed,
            )).hexdigest(), 16))
        print('Selecting subset of unlabelled data:')
        subset_path = '%s/subset-%02d.conllu' %(opt_workdir, training_round)
        unlabelled_subset = get_subset(
            unlabelled_data, opt_subset_size, random, opt_subset_attempts,
            with_replacement = False,
            diversify_attempts = opt_diversify_attempts,
            disprefer = previously_picked,
            sentence_modifier = drop_all_targets,
            write_file = subset_path
        )
        for d_index in unlabelled_subset.indices():
            try:
                previously_picked[d_index] += 1
            except KeyError:
                previously_picked[d_index] = 1

        print('Making predictions:')

        predictions = make_predictions(
            models, unlabelled_subset,
            training_round = training_round,
            opt_workdir = opt_workdir,
            # dataset_name = 'subset',    # TODO: add name when renames files ready
            filename_extension = filename_extension,
            opt_continue = opt_continue,
            opt_manually_predict = opt_manually_predict,
        )
        if opt_quit_after_prediction:
            print('\n*** Manual intervention requested. ***\n')
            print(
                'The predictions are ready. As --quit-after-predictions was'
                ' specified, you can now inspect and/or modify the predictions'
                ' and then continue tri-training by re-running this script'
                ' with the same settings and --continue.'
            )
            sys.exit(0)

        print('Teaching:')

        # TODO: provide options to control column weights;
        # for now, any difference triggers a disagreement
        column_weights = len(target_columns) * [1.0]
        new_datasets.append([])
        prediction_sets = []
        for learner_index in range(opt_learners):
            new_datasets[training_index].append(dataset_module.new_empty_set())
            prediction_sets.append(basic_dataset.load_or_map_from_filename(
                dataset_module.new_empty_set(),
                predictions[learner_index][1]
            ))
        event_counter = {}
        for subset_index in range(len(unlabelled_subset)):
            sentence_predictions = []
            for learner_index in range(opt_learners):
                sentence_predictions.append(prediction_sets[learner_index][subset_index])
            learner_index, merged_prediction = knowledge_transfer(
                sentence_predictions,
                target_columns, column_weights, opt_learners,
                #opt_max_teacher_disagreement_fraction,  # TODO: provide options for these
                #opt_min_teacher_agreements,
                #opt_learner_must_disagree,
                #opt_min_learner_disagreement,
                event_counter = event_counter,
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
            #       Note that masking the agreements with
            #       the learner's prediction with '_' is
            #       likely to result in training data with
            #       mostly masked predictions.

            # add new sentence to data set of learner
            new_datasets[training_index][learner_index].append(merged_prediction)
        print_event_counter(event_counter)

        new_training_sets = []
        for learner_index in range(opt_learners):
            if opt_init_seed:
                random.seed(int(hashlib.sha512('New dataset %d %d %s' %(
                    training_round, learner_rank, opt_init_seed,
                )).hexdigest(), 16))
            learner_rank = learner_index + 1
            new_dataset = new_datasets[training_index][learner_index]
            # write new labelled data to file
            tr_data_filename = '%s/new-candidate-set-%02d-%d.conllu' %(opt_workdir, training_round, learner_rank)
            f_out = open(tr_data_filename, 'w')
            new_dataset.save_to_file(f_out)
            f_out.close()
            new_size = new_dataset.get_number_of_items()
            print('Size of new dataset: %d items in %d sentences' %(
                new_size, len(new_dataset)
            ))
            if new_size > opt_augment_size:
                print('Pruning new dataset to augment size', opt_augment_size)
                new_datasets[training_index][learner_index] = get_subset(
                    new_dataset, opt_augment_size, random,
                    opt_augment_attempts, with_replacement = False,
                    prefer_smaller = True,
                    write_file = \
                    '%s/new-selected-set-%02d-%d.conllu' %(
                        opt_workdir, training_round, learner_rank
                    )
                )
            # compile training set for this iteration and learner
            # according to --last-k, --decay and --oversample
            if opt_last_k:
                # cannot use more than the available sets
                last_k = min(training_round, opt_last_k)
            else:
                last_k = training_round
            last_k_datasets = []
            for k in range(last_k):
                t_index = training_index - k
                weight = opt_last_decay ** k
                target_size = int(0.5 + weight * opt_augment_size)
                new_dataset = new_datasets[t_index][learner_index]
                if (
                    not k # never prune the dataset of the current round
                    and new_dataset.get_number_of_items() > target_size
                ):
                    new_dataset = get_subset(
                        new_dataset, target_size, random,
                        opt_last_decay_attempts, with_replacement = False,
                        write_file = \
                        '%s/new-decayed-set-%02d-%d-%02d.conllu' %(
                            opt_workdir, training_round, learner_rank, t_index
                        )
                    )
                last_k_datasets.append(new_dataset)
            last_k_datasets = basic_dataset.Concat(last_k_datasets)
            # add seed set
            seed_dataset = seed_sets[learner_index]
            if opt_oversample:
                # oversample seed data to match size of last k data
                target_size = last_k_datasets.get_number_of_items()
                seed_size = seed_dataset.get_number_of_items()
                if target_size > seed_size:
                    seed_dataset = get_subset(
                        seed_dataset, target_size, random,
                        with_replacement = False,
                    )
            new_training_set = basic_dataset.Concat(
                [seed_dataset, last_k_datasets],
                # replace blank labels with random labels
                sentence_modifier = basic_dataset.SentenceCompleter(
                    # use a new sentence completer in each round
                    # to make its random choices independent of
                    # previous rounds
                    random, target_columns, target_labelsets
                )
            )
            write_dataset(
                new_training_set,
                '%s/new-training-set-%02d-%d.conllu' %(
                    opt_workdir, training_round, learner_rank
                )
            )
            new_training_sets.append(new_training_set)

        print('Training of new models:')
        models = train_models(
            opt_learners, new_training_sets, epoch_selection_sets, model_modules,
            opt_model_init_type, opt_init_seed, training_round,
            opt_workdir, opt_manually_train, opt_continue,
            opt_verbose,
            monitoring_datasets = monitoring_datasets
        )

        print('Evaluating new models:')
        evaluate(
            models,
            dev_sets + unl_dev_sets,
            test_sets + unl_test_sets,
            opt_labelled_ids + opt_unlabelled_ids,
            training_round = training_round,
            filename_extension = filename_extension,
            opt_workdir = opt_workdir,
            opt_continue = opt_continue,
            opt_verbose = opt_verbose
        )


    print('\n== Final Model ==\n')
    # TODO

def make_predictions(
    models, dataset,
    training_round = 0,
    opt_workdir = './',
    dataset_name = '',
    filename_extension = '.data',
    opt_continue = False,
    opt_manually_predict = False,
    opt_verbose = False,
):
    ''' makes predictions for the given dataet for all learners
    '''
    manual_prediction_needed = []
    predictions = []
    if dataset_name:
        dataset_name = dataset_name + '-'
    for learner_index, model in enumerate(models):
        learner_rank = learner_index+1
        print('Learner:', learner_rank)
        model_fingerprint, model_path, model_module = model
        prediction_fingerprint = get_prediction_fingerprint(
             model_fingerprint, dataset
        )
        if opt_verbose:
            print('Prediction input and model fingerprint (shortened):', prediction_fingerprint[:40])
        prediction_path = '%s/prediction-%02d-%d-%s%s%s' %(
                opt_workdir, training_round, learner_rank,
                dataset_name,
                prediction_fingerprint[:20], filename_extension
        )
        print('Prediction output path:', prediction_path)
        if os.path.exists(prediction_path):
            if not opt_continue:
                raise ValueError(
                   'Conflicting prediction %r found. Use option'
                   ' --continue to re-use it.' %prediction_path
                )
            print('Re-using existing prediction')
        elif opt_manually_predict:
            # we will ask the user to predict the models when details
            # for all leaners have been printed
            manual_prediction_needed.append(learner_rank)
        else:
            # ask model module to predict the model
            model_module.predict(model_path, dataset.filename, prediction_path)
        predictions.append((prediction_fingerprint, prediction_path))
    if manual_prediction_needed:
        print('\n*** Manual prediction requested. ***\n')
        print(
            'Please make predictions for learner(s) %r using the details'
            ' above and the new files provided.\n' %manual_prediction_needed
        )
        sys.exit(0)
    return predictions

def evaluate(
    models, dev_sets, test_sets, set_names,
    training_round = 0,
    opt_workdir = './',
    filename_extension = '.data',
    opt_continue = False,
    opt_verbose  = False,
):
    for set_list, suffix, names in [
        (dev_sets,  '-dev',  set_names),
        (test_sets, '-test', set_names),
    ]:
        for d_index, dataset in enumerate(set_list):
            if dataset is None:
                continue
            name = names[d_index] + suffix
            predictions = make_predictions(
                models, dataset,
                training_round = training_round,
                opt_workdir = opt_workdir,
                dataset_name = name,
                filename_extension = filename_extension,
                opt_continue = opt_continue,
                opt_manually_predict = False,
                opt_verbose = opt_verbose,
            )
            gold_path = dataset.filename
            for learner_index, model in enumerate(models):
                learner_rank = learner_index + 1
                print('Evaluating learner %d on %s:' %(learner_rank, name))
                model_fingerprint, model_path, model_module = model
                prediction_fingerprint, prediction_path = predictions[learner_index]
                model_module.evaluate(
                    prediction_path, gold_path
                )

def hex2base62(h):
    s = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    i = int(h, 16)
    if not i:
        return '0'
    digits = []
    while i:
        d = i % 62
        digits.append(s[d])
        i = int(i/62)
    return ''.join(digits)

def get_prediction_fingerprint(model_fingerprint, unlabelled_subset, verbose = False):
    data_fingerprint = unlabelled_subset.hexdigest()
    if verbose:
        print('Prediction input data fingerprint (shortened):', data_fingerprint[:40])
    fingerprint = hashlib.sha512('%s:%s' %(
        model_fingerprint, data_fingerprint
    )).hexdigest()
    return hex2base62(fingerprint)

def get_model_fingerprint(model_init_seed, seed_set, epoch_selection_set = None, verbose = False):
    data_fingerprint = seed_set.hexdigest()
    if verbose:
        print('Seed data fingerprint (shortened):', data_fingerprint[:40])
    if epoch_selection_set:
        epoch_selection_fingerprint = epoch_selection_set.hexdigest()
    else:
        epoch_selection_fingerprint = 'N/A'
    if verbose:
        print('Epoch selection fingerprint (shortened):', epoch_selection_fingerprint[:40])
    model_fingerprint = hashlib.sha512('%s:%s:%s' %(
        model_init_seed, data_fingerprint, epoch_selection_fingerprint
    )).hexdigest()
    return hex2base62(model_fingerprint)

def get_model_selection_dataset(
    selection_type, dev_sets, seed_set, seed_set_10,
    size_limit = None, attempts = 5, rng = None,
    write_file = None
):
    if selection_type == 'last':
        return None
    if selection_type in ('dev', 'dev+remaining'):
        dev_set = basic_dataset.Concat(dev_sets)
    if selection_type in ('remaining', 'dev+remaining'):
        remaining = get_remaining(seed_set, rng)
    if selection_type == 'dev':
        retval = dev_set
    if selection_type == 'remaining':
        retval = remaining
    if selection_type == 'dev+remaining':
        retval = basic_dataset.Concat([dev_set, remaining])
    if selection_type == '9010':
        retval = seed_set_10
    if size_limit:
        retval_size = retval.get_number_of_items()
        size_limit = adjust_size(size_limit, retval_size)
        if retval_size > size_limit:
            retval = get_subset(
                retval, size_limit, rng, attempts, with_replacement = False,
                prefer_smaller = True
            )
    if write_file:
        write_dataset(retval, write_file)
    return retval

def get_remaining(dataset, rng, write_file = None):
    ''' dataset must be a Sample instance '''
    retval = dataset.clone()
    retval.set_remaining(rng)
    if write_file:
        write_dataset(retval, write_file)
    return retval

def get_subset(
    dataset, target_size, rng, attempts = 5, with_replacement = True,
    write_file = None, prefer_smaller = False,
    sentence_modifier = None,
    diversify_attempts = 1,
    disprefer = {},
):
    candidates = []
    ds_size = dataset.get_number_of_items()
    for _ in range(attempts):
        n_sentences = int(0.5 + len(dataset) * target_size / ds_size)
        candidate = basic_dataset.Sample(
            dataset, rng, n_sentences,
            with_replacement = with_replacement,
            sentence_modifier = sentence_modifier,
            diversify_attempts = diversify_attempts,
            disprefer = disprefer
        )
        size = candidate.get_number_of_items()
        deviation = abs(size - target_size)
        if size <= target_size or not prefer_smaller:
            priority = 0
        else:
            priority = 1
        candidates.append((priority, deviation, rng.random(), candidate))
    candidates.sort()
    retval = candidates[0][-1]
    if write_file:
        write_dataset(retval, write_file)
    return retval

def write_dataset(dataset, filename):
    f_out = open(filename, 'w')
    dataset.save_to_file(f_out)
    dataset.filename = filename
    f_out.close()

def get_disagreement(prediction1, prediction2, target_columns, column_weights):
    len1 = len(prediction1)
    len2 = len(prediction2)
    if len1 != len2:
        raise ValueError('trying to compare predictions with different lengths')
    disagreements = 0
    for item_index in range(len1):
        row1 = prediction1[item_index]
        row2 = prediction2[item_index]
        weighted_disagreement = 0.0
        for c_index, column in enumerate(target_columns):
            if row1[column] != row2[column]:
                weighted_disagreement += column_weights[c_index]
        if weighted_disagreement >= 1.0:
            disagreements += 1
    return disagreements

def merge_predictions(predictions, target_columns):
    first_prediction = predictions[0]
    len1 = len(first_prediction)
    remaining_predictions = predictions[1:]
    for pred in remaining_predictions:
        if len(pred) != len1:
            raise ValueError('trying to merge predictions with different lengths')
    retval = None
    for item_index in range(len1):
        for column in target_columns:
            first_label = first_prediction[item_index][column]
            all_agree = True
            for pred in remaining_predictions:
                if first_label != pred[item_index][column]:
                    all_agree = False
                    break
            if not all_agree:
                # found a disagreement
                # --> can no longer return first prediction
                if retval is None:
                    retval = first_prediction.clone()
                # blank label
                retval.unset_label(item_index, column)
    if not retval:
        return first_prediction
    else:
        return retval

def print_event_counter(event_counter):
    for key in sorted(event_counter.keys()):
        print(key, event_counter[key])

def count_event(event_counter, event):
    if event_counter is None:
        return
    try:
        event_counter[event] += 1
    except KeyError:
        event_counter[event] = 1

def knowledge_transfer(
    predictions, target_columns, column_weights, learners = 3,
    max_teacher_disagreement_fraction = 0.0,
    min_teacher_agreements = 2,
    learner_must_disagree = True, min_learner_disagreement = 1,
    verbose = False, event_counter = None
):
    # knowledge transfer candidates: first element says how much the teachers disagree
    kt_candidates = []
    s_length = float(len(predictions[0]))
    if verbose: print('Sentence length', s_length)
    for learner_index in range(learners):
        learner_prediction = predictions[learner_index]
        if learners == 1:
            t1_indices = [None]
        else:
            t1_indices = range(learners)
        for teacher1_index in t1_indices:
            if learners > 1:
                # co- or tri-training: teacher cannot
                # also be the learner
                if teacher1_index == learner_index:
                    continue
                t1_prediction = predictions[teacher1_index]
            if learners <= 2:
                # self- or co-training: no second teacher
                t2_indices = [None]
            else:
                # tri-training
                t2_indices = range(teacher1_index+1, learners)
            for teacher2_index in t2_indices:
                if learners == 1:
                    # self-training
                    teachers_predictions = [learner_prediction]
                    if verbose: print('Learner %d, teacher %d' %(
                        learner_index+1, teacher1_index+1
                    ))
                elif learners == 2:
                    # co-training
                    teachers_predictions = [t1_prediction]
                    if verbose: print('Learner %d, teacher %d' %(
                        learner_index+1, teacher1_index+1
                    ))
                else:
                    if teacher2_index == learner_index:
                        continue
                    if verbose: print('Learner %d, teachers %d and %d' %(
                        learner_index+1, teacher1_index+1, teacher2_index+1
                    ))
                    t2_prediction = predictions[teacher2_index]
                    # measure disagreement between teachers
                    teacher_disagreement = get_disagreement(
                        t1_prediction, t2_prediction,
                        target_columns, column_weights
                    )
                    if verbose: print('Teacher disagreement:', teacher_disagreement)
                    teacher_disagreement_fraction = teacher_disagreement / s_length
                    if teacher_disagreement_fraction > max_teacher_disagreement_fraction:
                        if verbose: print(teacher_disagreement_fraction, 'exceeds max_teacher_disagreement_fraction')
                        continue
                    if s_length - teacher_disagreement < min_teacher_agreements:
                        if verbose: print(s_length - teacher_disagreement, 'below min_teacher_agreements')
                        continue
                    teachers_predictions = [
                        t1_prediction, t2_prediction
                    ]
                # optionally consider disagreement with learner
                if learner_must_disagree:
                    merged_prediction = merge_predictions(
                        teachers_predictions, target_columns
                    )
                    learner_disagreement = get_disagreement(
                        learner_prediction, merged_prediction,
                        target_columns, column_weights
                    )
                    if learner_disagreement < min_learner_disagreement:
                        # skip this candidate as the learner
                        # is unlikely to learn much from it
                        if verbose: print('below min_learner_disagreement')
                        continue
                else:
                    # postpone merging to when it is needed
                    merged_prediction = None
                    learner_disagreement = 0
                # record candidate
                priority = (teacher_disagreement, -learner_disagreement)
                kt_candidates.append((
                    priority, random.random(),
                    learner_index, teacher1_index, teacher2_index,
                    merged_prediction
                ))
    count_event(event_counter, ('n_kt_candidates', len(kt_candidates)))
    if not kt_candidates:
        # can happen when predictions agree and learner_must_disagree is set
        if verbose:
            print('No candidates')
        return -1, None
    kt_candidates.sort()
    if verbose:
        print('Number of candidates:', len(kt_candidates))
    _, _, learner_index, t1_index, t2_index, prediction = kt_candidates[0]
    if t2_index is None:
        count_event(event_counter, ('kt from', t1_index+1, 'to', learner_index+1))
    else:
        count_event(event_counter, ('kt from', t1_index+1, 'and', t2_index+1, 'to', learner_index+1))
    if prediction is None:
        t1_prediction = predictions[t1_index]
        t2_prediction = predictions[t2_index]
        prediction = merge_predictions(
            [t1_prediction, t2_prediction],
            target_columns
        )
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

def train_models(
    opt_learners, training_sets, epoch_selection_sets, model_modules,
    opt_model_init_type, opt_init_seed, training_round,
    opt_workdir, opt_manually_train, opt_continue,
    opt_verbose,
    monitoring_datasets = [],
):
    retval = []
    manual_training_needed = []
    for learner_index in range(opt_learners):
        learner_rank = learner_index+1
        print('Learner:', learner_rank)
        model_init_seed = get_model_seed(
            opt_model_init_type, opt_init_seed, learner_rank, training_round,
        )
        print('Model initialisation seed:', model_init_seed)
        training_set = training_sets[learner_index]
        epoch_selection_set = epoch_selection_sets[learner_index]
        model_fingerprint = get_model_fingerprint(
            model_init_seed, training_set, epoch_selection_set,
            verbose = opt_verbose
        )
        if opt_verbose:
            print('Model fingerprint (shortened):', model_fingerprint[:40])
        model_path = '%s/model-%02d-%d-%s' %(
                opt_workdir, training_round,
                learner_rank, model_fingerprint[:20]
        )
        print('Model path:', model_path)
        # choose model for learner
        model_module = model_modules[learner_index % len(model_modules)]
        if os.path.exists(model_path):
            if not opt_continue:
                raise ValueError(
                   'Conflicting model %r found. Use option'
                   ' --continue to re-use it.' %model_path
                )
            print('Re-using existing model')
        elif opt_manually_train:
            # we will ask the user to train the models when details
            # for all leaners have been printed
            manual_training_needed.append(learner_rank)
        else:
            # ask model module to train the model
            model_module.train(
                training_set.filename, model_init_seed, model_path,
                epoch_selection_set, monitoring_datasets,
            )
        retval.append((model_fingerprint, model_path, model_module))
    if manual_training_needed:
        print('\n*** Manual training requested. ***\n')
        print(
            'Please train models for learner(s) %r using the details'
            ' above and the new files provided.\n' %manual_training_needed
        )
        sys.exit(0)
    return retval


if __name__ == "__main__":
    main()

