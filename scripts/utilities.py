#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

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

