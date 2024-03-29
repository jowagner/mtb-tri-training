#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

from __future__ import print_function

import hashlib
import random
import sys

import basic_dataset
import conllu_dataset

def print_usage():
    print('Usage: %s [options]' %(os.path.split(sys.argv[0])[-1]))
    print("""

Processes conllu data from stdin to stdout, removing all comments.

Options:

    --min-length  NUMBER    Only copy sentences with at least NUMBER tokens,
                            not counting tokens with an index containing
                            '.' or '-'.
                            (Default: 5)

    --max-length  NUMBER    Only copy sentences with at most NUMBER tokens,
                            not counting tokens with an index containing
                            '.' or '-'.
                            (Default: 40)

    --min-chars  NUMBER     Only copy sentences with at least NUMBER characters,
                            not counting whitespace.
                            (Default: 1)

    --max-chars  NUMBER     Only copy sentences with at most NUMBER characters,
                            not counting whitespace.
                            (Default: 25000)

    --keep-comments         Don't remove comments.
                            (Default: lines starting with '#' will not be
                            copied)

    --fraction  NUMBER      In addition to other filters, stochastically
                            only keep a fraction of the sentences.
                            (Default: 1.0 = no stochastic dropping of
                            sentences)

    --init-seed  STRING     Initialise random number generator for stochastic
                            filtering with STRING.
                            With an empty string as seed, a system seed will
                            be used.
                            (Default: 42)

    --info  STRING          No change of behaviour. Can be used to add
                            information to the command line viewed e.g. in
                            top/htop.
""")

def main():
    opt_help = False
    opt_verbose = False
    opt_debug   = False
    opt_min_length = 5
    opt_max_length = 40
    opt_min_chars = 1
    opt_max_chars = 25000
    opt_keep_comments = False
    opt_skip = 0.0
    opt_init_seed = '42'
    while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
        option = sys.argv[1]
        option = option.replace('_', '-')
        del sys.argv[1]
        if option in ('--help', '-h'):
            opt_help = True
            break
        elif option == '--info':
            del sys.argv[1]
        elif option == '--min-length':
            opt_min_length = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--max-length':
            opt_max_length = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--min-chars':
            opt_min_chars = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--max-chars':
            opt_max_chars = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--keep-comments':
            opt_keep_comments = False
        elif option == '--fraction':
            opt_skip = 1.0 - float(sys.argv[1])
            del sys.argv[1]
        elif option == '--init-seed':
            opt_init_seed = sys.argv[1]
            del sys.argv[1]
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

    if opt_init_seed:
        # For why using sha512, see Joachim's answer on
        # https://stackoverflow.com/questions/41699857/initialize-pseudo-random-generator-with-a-string
        random.seed(int(hashlib.sha512(opt_init_seed).hexdigest(), 16))

    columns_to_blank = conllu_dataset.get_target_columns()
    blank_columns = basic_dataset.SentenceDropout(
        random, 
        target_columns = columns_to_blank,
        dropout_probabilities = len(columns_to_blank) * [1.0],
    )
    length_filter = basic_dataset.SentenceFilter(
        [],                              # no specific target columns
        min_length = opt_min_length,
        max_length = opt_max_length,
        min_chars  = opt_min_chars,
        max_chars  = opt_max_chars,
        skip_prob  = opt_skip,
        rng        = random,
    )
    while True:
        conllu = conllu_dataset.ConlluDataset()
        sentence = conllu.read_sentence(sys.stdin)
        if sentence is None:
            break
        conllu.append(sentence)
        conllu.save_to_file(
            sys.stdout,
            sentence_filter    = length_filter,
            sentence_completer = blank_columns,
            remove_comments    = not opt_keep_comments,
        )

if __name__ == "__main__":
    main()

