#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import random
import sys

import conllu_dataset

def print_usage():
    print('Usage: %s [options]' %(os.path.split(sys.argv[0])[-1]))
    print("""

Processes conllu data from stdin to stdout, removing all comments.

Options:

    --min-length  NUMBER    Only copy sentences with at least NUMBER tokens
                            (not containing tokens with an index containing
                            '.' or '-')
                            (Default: 5)

    --max-length  NUMBER    Only copy sentences with at most NUMBER tokens
                            (not containing tokens with an index containing
                            '.' or '-')
                            (Default: 40)

    --keep-comments         Don't remove comments.
                            (Default: lines starting with '#' will not be
                            copied)

    --fraction  NUMBER      In addition to other filters, stochastically
                            only keep a fraction of the sentences.
                            (Default: 1.0 = no stochastic dropping of
                            sentences)

    --init-seed  STRING     Initialise random number generator for stochastic
                            filtering with STRING.
                            (Default: use system seed)
""")

def main():
    opt_help = False
    opt_verbose = False
    opt_debug   = False
    opt_min_length = 5
    opt_max_length = 40
    opt_keep_comments = False
    opt_skip = 0.0
    opt_init_seed = None
    while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
        option = sys.argv[1]
        option = option.replace('_', '-')
        del sys.argv[1]
        if option in ('--help', '-h'):
            opt_help = True
            break
        elif option == '--min-length':
            opt_min_length = sys.argv[1]
            del sys.argv[1]
        elif option == '--max-length':
            opt_max_length = sys.argv[1]
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
        len(columns_to_blank) * [1.0],   # dropout probabilities
    )
    length_filter = basic_dataset.SentenceFilter(
        [],                              # no specific target columns
        min_length = opt_min_length,
        max_length = opt_max_length,
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
            skip_prob          = opt_skip,
            rng                = random.random,
        )

if __name__ == "__main__":
    main()

