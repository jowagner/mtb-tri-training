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

def print_usage():
    print('Usage: ls data-ichec/*.tsv | %s [options]' %(sys.argv[0].split('/')[-1]))
    print()
    print('Writes to folders tbweights, te-parse, te-worker and te-combine')
    print("""
Options:

    --test-type  STRING     whether to test on the dev or test section of each
                            treebank (default: dev)

    --collection  STRING    append data set collection STRING to the list of
                            collections; STRING must be a space- or
                            colon-separated list of data set IDs (if the list
                            is empty, the UD v2.3 Czech, English and French
                            collections with non-pud treebanks that do not
                            withhold surface tokens will be used)

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
                                    trainer rank (1-3) and training round
                                )
                            (default: do not provide the models with a seed,
                            usually meaning that they use a system seed)

""")

opt_help  = False
test_type = 'dev'
opt_debug = False
opt_seed  = None

import sys
while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
    option = sys.argv[1]
    del sys.argv[1]
    if option in ('--help', '-h'):
        opt_help = True
        break
    elif option == '--test-type':
        test_type = sys.argv[1]
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

import hashlib
if opt_seed:
    random.seed(int(hashlib.sha512(opt_seed).hexdigest(), 16))


