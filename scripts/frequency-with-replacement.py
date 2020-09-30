#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import random
import sys

repetitions = 100000
items       = 10000

data = list(range(items))

freq2count = items * [0]
for repetition in range(repetitions):
    sample = [random.choice(data) for _ in data]
    item2freq = items * [0]
    for item in sample:
        item2freq[item] += 1
    for freq in item2freq:
        freq2count[freq] += 1

total1 = 0
total2 = repetitions*items
for freq, count in enumerate(freq2count):
    if count:
        total1 += count
        print '%3d\t%7.3f\t%7.3f\t%7.3f\t%d' %(
           freq,
           100.0*count/float(items*repetitions),
           100.0*total1/float(items*repetitions),
           100.0*total2/float(items*repetitions),
           count
        )
        total2 -= count

