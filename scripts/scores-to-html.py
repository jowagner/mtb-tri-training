#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

from __future__ import print_function

import sys

print('<html><body>')

legend = """
<p>black bold = above or matching best baseline for this language and parser
<p>red bold = below this seed and method's baseline
<p>blue highlight = 0.5 to 1.0 above this seed and method's baseline
<p>green highlight = 1.0 to 2.0 above this seed and method's baseline
<p>yellow highlight = at least 2.0 above this seed and method's baseline
<p>grey highlight = worst baseline for this language and parser
</p>
"""

header = sys.stdin.readline().split()

langandparser2bestbaseline = {}
langandparser2worstbaseline = {}

baseline_column = header.index('0')
learner_column  = header.index('Learner')
testset_column  = header.index('TestSetIndex')

rows = []
while True:
    line = sys.stdin.readline()
    if not line:
        break
    row = line.split()
    if row[learner_column] != 'Ensemble':
        continue
    if row[testset_column] != '0':
        continue
    rows.append(row)
    language = row[0]
    parser   = row[1]
    key = (language, parser)
    baselinescore = float(row[baseline_column])
    try:
        bestscore = langandparser2bestbaseline[key]
    except:
        bestscore = -1.0
    if baselinescore > bestscore:
        langandparser2bestbaseline[key] = baselinescore
    try:
        worstscore = langandparser2worstbaseline[key]
    except:
        worstscore = 9999.0
    if baselinescore < worstscore:
        langandparser2worstbaseline[key] = baselinescore

rows.sort()
print('<p>%d rows</p>' %len(rows))

rows.append((None,None))

last_language = None
last_parser   = None

l2text = {
    'e': 'English',
    'h': 'Hungarian',
    'u': 'Uyghur',
    'v': 'Vietnamese',
}
p2text = {
    'f': 'udpipe-future',
    'g': '+fasttext',
    'h': '+elmo',
}

for row in rows:
    parser = row[1]
    language = row[0]
    if last_parser != parser and last_parser is not None:
        print('</table>')
    if last_language != language and last_language is not None:
        print(legend)
    if parser is None:
        break
    if last_language != language:
        print('<h2>%s</h2>' %l2text[language])
    if last_parser != parser:
        print('<h3>%s</h3>' %p2text[parser])
        print('<table cellpadding="4" border="1">')
    print('<tr>')
    for i in range(2,baseline_column-5):
        print('<td>%s</td>' %row[i])
    # number of rounds
    print('<td>%s</td>' %row[baseline_column-2])
    # size in last round
    print('<td>%.1fk</td>' %(0.001*int(row[baseline_column-1])))
    bestbaselinescore = langandparser2bestbaseline[(language, parser)]
    worstbaselinescore = langandparser2worstbaseline[(language, parser)]
    score = float(row[baseline_column])
    sccode = '<div title="%.9f">%.1f</div>' %(score, score)
    if score == bestbaselinescore:
        print('<td><b>%s</b></td>' %sccode)
    elif score == worstbaselinescore:
        print('<td bgcolor="#c5c5c5">%s</td>' %sccode)
    else: 
        print('<td>%s</td>' %sccode)
    rowbaseline = score
    for text in row[baseline_column+1:]:
        score = float(text)
        if score >= rowbaseline + 2.0:
            tdcode = '<td bgcolor="#ffff00">'
        elif score >= rowbaseline + 1.0:
            tdcode = '<td bgcolor="#e2f09f">'
        elif score >= rowbaseline + 0.5:
            tdcode = '<td bgcolor="#d2daf3">'
        else:
            tdcode = '<td>'
        sccode = '<div title="%.9f">%.1f</div>' %(score, score)
        if score >= bestbaselinescore:
            print('%s<b>%s</b></td>' %(tdcode, sccode))
        elif score < rowbaseline:
            print('%s<b><font color="red">%s</font></b></td>' %(tdcode, sccode))
        else:
            print('%s%s</td>' %(tdcode, sccode))
    last_parser = parser
    last_language = language

print('<body></html>')
        
example_input = """
Language	Parser	AugmentSizeIndex	Method	NumberOfLearners	Learner	TestSetIndex	Rounds	0	1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19
e	f	0	-x--	3	Ensemble	0	4	75.691903929	75.922538572	76.280419914	76.336090345	76.606489582
e	f	0	-x--	3	Ensemble	1	4	76.119700351	76.677558177	76.972425885	77.243385400	77.064073956
e	f	0	-x--	3	L1	0	4	75.341975505	75.389693017	75.795291872	75.870844600	76.045808812
e	f	0	-x--	3	L1	1	4	75.450270960	76.219317820	76.394644565	77.056104558	76.478323239
e	f	0	-x--	3	L2	0	4	75.218705265	75.250516940	75.493080961	75.592492445	76.105455702
e	f	0	-x--	3	L2	1	4	75.673414090	76.203379025	76.302996493	76.514185528	76.964456487
e	f	0	-x--	3	L3	0	4	75.234611102	75.668045173	75.998091300	76.415619532	76.252584699
e	f	0	-x--	3	L3	1	4	75.908511317	76.326904686	76.565986611	76.721389863	76.558017214
e	f	0	ox--	3	Ensemble	0	7	75.588515985	75.962303165	76.507078098	76.638301257	76.526960394	76.519007476	76.678065850	76.908700493
"""
