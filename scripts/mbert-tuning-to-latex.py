#!/usr/bin/env python

from __future__ import print_function

import sys

key2data = {}
layers = set()
poolings = set()
languages = set()
lang = '?'
while True:
    line = sys.stdin.readline()
    if not line:
        break
    fields = line.split()
    if line.startswith('='):
        if len(fields) == 3:
            lang = fields[1]
        elif 'all languages' in line:
            lang = 'all'
        else:
            lang = '?'
    else:
        assert len(fields) == 6
        assert fields[3] == 'samples,'
        assert fields[5] == 'stddev)'
        kfields = fields[0].split('-')
        score = float(fields[1])
        samples = int(fields[2][1:])
        stddev = float(fields[4])
        if len(kfields) == 4 and kfields[1][0] == 'L':
            pooling = kfields[0]
            layer   = kfields[1] + '-' + kfields[2]
            lang    = kfields[3]
        elif len(kfields) == 3 and kfields[0][0] == 'L':
            pooling = 'all'
            layer   = kfields[0] + '-' + kfields[1]
            lang    = kfields[2]
        elif len(kfields) == 3 and kfields[0][0] in ('aflmz'):
            pooling = kfields[0]
            layer   = kfields[1] + '-' + kfields[2]
            lang    = 'all'
        elif len(kfields) == 2 and kfields[0][0] == 'L':
            pooling = 'all'
            layer   = kfields[0] + '-' + kfields[1]
            lang    = 'all'
        elif len(kfields) == 1 and kfields[0][0] in ('aflmz'):
            pooling = kfields[0]
            layer = 'all'
            assert lang != '?'   # language comes from section heading
        else:
            raise ValueError('cannot handle key %r in line %r' %(kfields, line))
        if layer[0] != 'L' and layer != 'all':
            raise ValueError('Read layer %r from line %r' %(layer, line))
        assert lang != '?'
        if lang == 'all':
            lang = 'zztotal'
        if pooling != 'all': poolings.add(pooling)
        if layer   != 'all': layers.add(layer)
        languages.add(lang)   
        key = (pooling, layer, lang)
        assert key not in key2data
        key2data[key] = (score, samples, stddev)

print(r"""
\begin{table*}
\centering
\begin{tabular}{l|%s}""" %((len(layers)+1) * 'r|'))

row = []
row.append('Pooling')
for layer in sorted(list(layers)):
    if layer.endswith('-0768'):
        row.append(layer[1:-5])
    elif layer.endswith('-1024'):
        row.append(layer[1:-5] + 'E') 
    else:
        raise ValueError('unsupported layer %s' %layer)
row.append('Average')
print(' & '.join(row), '\\\\')

for lang in sorted(list(languages)):
    print(r'\multicolumn{%d}{l|}{%s} \\' %(
        len(layers)+2, lang
    ))
    print('\\hline')
    for pooling in sorted(list(poolings)):
        row = []
        row.append(pooling)
        for layer in sorted(list(layers)):
            key = (pooling, layer, lang)
            score, samples, stddev = key2data[key]
	    row.append('%.1f' %score)
        key = (pooling, 'all', lang)
        score, samples, stddev = key2data[key]
        row.append('%.1f' %score)
        print(' & '.join(row), '\\\\')
    print('\\hline')
    pooling = 'all'
    row = []
    row.append('Average')
    for layer in sorted(list(layers)):
        key = (pooling, layer, lang)
        score, samples, stddev = key2data[key]
        row.append('%.1f' %score)
    key = (pooling, 'all', lang)
    try:
        score, samples, stddev = key2data[key]
        row.append('%.1f' %score)
    except KeyError:
        row.append('--')
    print(' & '.join(row), '\\\\')
    print('\\hline')
    
print(r"""\hline
\end{tabular}
\caption{Development set LAS for training UDPipe-Future with word embeddings taken from
different BERT layers.}
\label{tab:bert-layers}
\end{table*}
""")

