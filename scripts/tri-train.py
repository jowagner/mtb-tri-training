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

import hashlib
import os
import random
import string
import sys

def print_usage():
    print('Usage: %s [options]' %(os.path.split(sys.argv[0])[-1]))
    print("""
Options:

    --test-type  STRING     whether to test on the dev or test section
                            (default: dev)

    --labelled  STRING
    --unlabelled  STRING    append data sets to the list of labelled or
                            unlabelled data sets; STRING must be a space- or
                            colon-separated list of data set IDs;
                            the module specified with --dataset-module
                            interprets these IDs to load the data

    --seed  STRING          initialise random number generator with STRING;
                            (default or empty string: use system seed)

    --model-seed  TYPE      derive a seed for each model to be trained from
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

    --oversample            Oversample seed data to match size of 

    --last-k  NUMBER        Only use the automatically labelled data of the
                            last k tri-training iterations

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
    opt_debug = False
    opt_seed  = None
    opt_labelled_ids = []
    opt_unlabelled_ids = []
    opt_dataset_module = 'conllu_dataset'
    opt_model_module   = 'uuparser_model'
    opt_model_modules  = []
    opt_model_seed     = None

    while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
        option = sys.argv[1]
        del sys.argv[1]
        if option in ('--help', '-h'):
            opt_help = True
            break
        elif option == '--test-type':
            test_type = sys.argv[1]
            del sys.argv[1]
        elif option == '--seed':
            opt_seed = sys.argv[1]
            del sys.argv[1]
        elif option == '--model-seed':
            opt_model_seed = sys.argv[1]
            del sys.argv[1]
        elif option in ('--labelled', '--unlabelled'):
            for tbid in sys.argv[1].replace(':', ' ').split():
                if option == '--labelled':
                    opt_labelled_ids.append(tbid)
                else:
                    opt_unlabelled_ids.append(tbid)
            del sys.argv[1]
        elif option == '--debug':
            opt_debug = True
        else:
            print('Unsupported option %s' %option)
            opt_help = True
            break

    if len(sys.argv) != 1:
        opt_help = True

    if opt_help:
        print_usage()
        sys.exit(0)

    if opt_seed:
        random.seed(int(hashlib.sha512(opt_seed).hexdigest(), 16))

    for training_round in range(5):
        for learner_rank in (1,2,3):
            print('%d\t%d\t%r' %(learner_rank, training_round,
                get_model_seed(opt_model_seed, opt_seed, learner_rank, training_round)))


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
    elif mode == 'combine':
        if main_seed is None:
            raise ValueError('cannot combine model seed without main seed')
        return '%s%d%d' %(main_seed, learner_rank, training_round)
    else:
        raise ValueError('unknown model seed type %r' %mode)


if __name__ == "__main__":
    main()

