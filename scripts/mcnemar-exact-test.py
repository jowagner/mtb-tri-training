#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# This implements the exact-P-value under the heading "Variations" on
# https://en.wikipedia.org/wiki/McNemar%27s_test
# (no citation provided)
# and we confirm empirically that this test matches a simple (but
# computationally expensive) randomisation test.
# TODO: Is this the binomial sign test and/or Liddell's exact test?

from __future__ import print_function

import random
import sys

a = 0
b = 0
c = 0
d = 0
data = []
while True:
    line = sys.stdin.readline()
    if not line:
        break
    row = line.rstrip()
    if row == 'C\tC':
        a += 1
    elif row == 'I\tC':
        b += 1
        data.append(row)
    elif row == 'C\tI':
        c += 1
        data.append(row)
    elif row == 'I\tI':
        d += 1
    else:
        raise ValueError('unexpected input')

print('confusion table:', a, b, c, d)

print('X2 =', (b-c)**2 / float(b+c))

# https://gist.github.com/rougier/ebe734dcc6f4ff450abf
# with suggestions from
# https://gist.github.com/keithbriggs
# and alisianoi's answer on
# https://stackoverflow.com/questions/26560726/python-binomial-coefficient
# (can be replaced with math.comb() when Python 3.8 or
# greater is used widely)

def binomial(n, k):
    if k < 0 or k > n:
        # for compatibility with scipy.special.{comb, binom}
        return 0
    if k == 0 or k == n:
        return 1
    b = 1
    for t in range(min(k, n - k)):
        b = (b * n) // (t + 1)
        n -= 1
    return b

if b < c:
    # swap values
    b,c = c,b
epv = 0
i = b
n = b+c
while i <= n:
    epv += binomial(n, i)
    i += 1

print('exact p-value as fraction = %d / %d' %(epv, 2**(n-1)))

is_reduced = False
while True:
    overflow = False
    try:
        epv_as_float = float(epv)
    except OverflowError:
        overflow = True
    f = 0.5 ** (n-1)
    if f > 0.0 and not overflow:
        if is_reduced:
            print('reduced fraction to %d / %d for conversion to float' %(epv, 2**(n-1)))
        epv = f * epv_as_float
        break
    else:
        n -= 1
        epv = epv >> 1
        is_reduced = True

print('exact p-value    =', epv)
print('exact p-value    = %.9f (rounded to 9 digits)' %epv)
print('exact p-value    = %.6f (rounded to 6 digits)' %epv)

# now perform a randomisation test
reps = 2500000
delta = abs(b - c)
p = 0
remaining = reps
done = 0
while remaining:
    b = 0
    c = 0
    for row in data:
        if random.random() < 0.5:
            if row == 'I\tC':
                b += 1
            elif row == 'C\tI':
                c += 1
            else:
                raise ValueError
        else:
            # swap order
            if row == 'I\tC':
                c += 1
            elif row == 'C\tI':
                b += 1
            else:
                raise ValueError
    s_delta = abs(b - c)
    if s_delta >= delta: # or (s_delta == delta and random.random() < 0.5):
        p += 1
    remaining -= 1
    done += 1
    if done and (done % 125000) == 0:
        print('empircal p-value = %.6f (n = %d)' %(p / float(done), done))

