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
import sys

def print_usage():
    print('Usage: %s [options]' %(os.path.split(sys.argv[0])[-1]))
    print("""
Options:

    --test-type  STRING     whether to test on the dev or test section
                            (default: dev)

    --labelled  STRING
    --unlabelled  STRING    append data sets to the list of labelled or
                            unlablled data sets; STRING must be a space- or
                            colon-separated list of data set IDs

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
                                    trainer rank (1-3) and training round,
                                    i.e. adding unique additional digits to the
                                    main seed
                            (default: do not provide the models with a seed,
                            usually meaning that they use a system seed)

    --model-module  NAME    train models using the module NAME
                            #(default: model_from_script)
                            (default: model_from_script)

    --model-script  FILE    training script to use with model_from_script
                            (default: train-

""")

opt_help  = False
test_type = 'dev'
opt_debug = False
opt_seed  = None

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

if opt_seed:
    random.seed(int(hashlib.sha512(opt_seed).hexdigest(), 16))



