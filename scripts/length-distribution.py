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

Processes conllu data from stdin and output sentence length distribution

Options:

    --min-length  NUMBER    Only count sentences with at least NUMBER tokens
                            (not containing tokens with an index containing
                            '.' or '-')
                            (Default: 0 = no limit)

    --max-length  NUMBER    Only count sentences with at most NUMBER tokens
                            (not containing tokens with an index containing
                            '.' or '-')
                            (Default: 0 = no limit)

    --bar-length  NUMBER    How many stars to print for the highest
                            frequency in the bar graph.
                            (Default: 80)
""")

def main():
    opt_help = False
    opt_verbose = False
    opt_debug   = False
    opt_min_length = 0
    opt_max_length = 0
    opt_bar_length = 80
    while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
        option = sys.argv[1]
        option = option.replace('_', '-')
        del sys.argv[1]
        if option in ('--help', '-h'):
            opt_help = True
            break
        elif option == '--min-length':
            opt_min_length = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--max-length':
            opt_max_length = int(sys.argv[1])
            del sys.argv[1]
        elif option == '--bar-length':
            opt_bar_length = int(sys.argv[1])
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

    max_length = opt_min_length
    counts = {}
    while True:
        conllu = conllu_dataset.ConlluDataset()
        sentence = conllu.read_sentence(sys.stdin)
        if sentence is None:
            break
        length = len(sentence)
        if opt_min_length and length < opt_min_length:
            continue
        if opt_max_length and length < opt_max_length:
            continue
        try:
            counts[length] += 1
        except:
            counts[length] = 1
        if length > max_length:
            max_length = length
    max_count = float(max(counts.values()))
    for length in range(opt_min_length, max_length+1):
        try:
            count = counts[length]
        except:
            count = 0
        bar = int(0.5+opt_bar_length*count/max_count)
        print('%d\t%d\t%s' %(
            length, count, bar * '*'
        ))

if __name__ == "__main__":
    main()

