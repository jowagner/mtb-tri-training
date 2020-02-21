#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import os

def hex2base62(h, min_length = 0):
    s = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    i = int(h, 16)
    if not i:
        return max(1, min_length) * '0'
    digits = []
    while i or (min_length and len(digits) < min_length):
        d = i % 62
        digits.append(s[d])
        i = int(i/62)
    return ''.join(digits)

def bstring(s):
    if type(b'') is str:
        return s
    if type(s) is bytes:
        return s
    return s.encode('utf-8')

def std_string(s):
    if type(b'') is str:
       return s
    if type(s) is bytes:
       return s.decode('utf-8')
    return s

def b_ord(s):
    if type(b'') is str:
        return ord(s)
    if type(s) is int:
        # probably called with binarystring[some_index]
        return s
    return s[0]   # in Python 3, accessing an element of a byte string yields an integer

def iteritems(d):
    try:
        return d.iteritems()
    except AttributeError:
        return d.items()

def makedirs(required_dir):
    if not os.path.exists(required_dir):
        try:
            os.makedirs(required_dir)
        except OSError:
            # folder was created by another process
            # between the to calls above
            # (in python 3, we will be able to use exist_ok=True)
            pass
