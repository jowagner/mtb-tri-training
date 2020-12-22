#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

from __future__ import print_function

import os
import sys

import utilities

def print_usage():
    print('Usage: %s [options] < main-comparison.tsv > test-results.tex' %(os.path.split(sys.argv[0])[-1]))
    print("""
Options:

    --models  NAMES         Colon- or space-separated list of models to evaluate
                            (default: udpf:elmo)

    --test-types  NAMES     Colon- or space-separated list of test set types
                            to include
                            (default: dev:test)

    --number-format  FORMAT  Format string to apply to LAS scores
                            (default: %.1f)

    --dev-placement  WHERE  Where to put the dev results:
                            "left" = on the left in one block
                            "next" = next to each test result
                            (Default: left)
""")

def main():
    opt_help  = False
    opt_debug  = False
    opt_verbose = False
    opt_models     = ('udpf', 'elmo')
    opt_test_types = ('dev', 'test')
    opt_dev_placement = 'left'
    opt_number_format = '%.1f'
    while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
        option = sys.argv[1]
        option = option.replace('_', '-')
        del sys.argv[1]
        if option in ('--help', '-h'):
            opt_help = True
            break
        elif option == '--number-format':
            opt_number_format = sys.argv[1]
            del sys.argv[1]
        elif option == '--dev-placement':
            opt_dev_placement = sys.argv[1]
            del sys.argv[1]
        elif option == '--models':
            opt_models = sys.argv[1].replace(':', ' ').split()
            del sys.argv[1]
        elif option == '--test-types':
            opt_test_types = sys.argv[1].replace(':', ' ').split()
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

    columns = []
    latex_header_1 = []
    latex_header_2 = []
    latex_header_3 = []
    columns.add('Language')
    latex_header_1.add('')
    latex_header_2.add('')
    latex_header_3.add('\\textbf{Lang.}')

    if opt_dev_placement == 'left' or len(opt_test_types) == 1:
        for test_type in opt_test_types:
            if len(opt_test_types) > 1:
                latex_header_1.add('\\multicolumn{%d}{c|}{\\textbf{%s}}' %(
                    2 * len(opt_models),
                    test_type.title(),
                ))
            for parser in opt_models:
                if len(opt_models) > 1:
                    latex_header_2.add('\\multicolumn{2}{c|}{\\textbf{%s}}' %parser)
                columns.add('b-%s-%s-LAS' %(parser, test_type))
                latex_header_3.add('\\textbf{B}')
                columns.add('tt-%s-%s-LAS' %(parser, test_type))
                latex_header_3.add('\\textbf{TT}')

    elif opt_dev_placement == 'next':
        for parser in opt_models:
            latex_header_1.add('\\multicolumn{%d}{c|}{\\textbf{%s}}' %(
                2 * len(test_types),
                parser,
            ))
            for tt_key, tt_text in [
                ('b', 'B')
                ('t', 'TT')
            ]:
                latex_header_2.add('\\multicolumn{%d}{c|}{\\textbf{%s}}' %(
                    len(opt_test_types),
                    tt_text
                ))
                for test_type in opt_test_types:
                    columns.add('%s-%s-%s-LAS' %(tt_key, parser, test_type))
                    latex_header_3.add('\\textbf{%s}' %(test_type.title()))
    else:
        raise ValueError('unsupported placement %s' %opt_dev_placement)

    print("""\\begin{table*}
\\centering
\\begin{tabular}{l%s}
""" %(2*len(opt_models)*len(opt_test_types)*'|r'))
    print((' & '.join(latex_header_1))+' \\\\')
    print((' & '.join(latex_header_2))+' \\\\')
    print((' & '.join(latex_header_3))+' \\\\')
    print('\\hline')
    header = sys.stdin.readline().rstrip().split('\t')
    lang_column = header.index('Language')
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        row = line.rstrip().split('\t')
        language = row[lang_column]
        if language == 'Average':
            break
        latex_row = []
        for column in columns:
            index = header.index(column)
            cell = row[index]
            if column == 'Language':
                latex_row.append(cell)
            else:
                latex_row.append(opt_number_format %float(cell))
        print((' & '.join(latex_row))+' \\\\')

if __name__ == "__main__":
    main()

