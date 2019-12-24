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

import os
import sys

print('<html><body>')

legend = """
<p>black bold = above or matching best baseline for this language and parser
<p>red bold = below this seed and method's baseline
<p>background dark purple: below worst baseline
<p>background dark grey: in bottom 2.5% of baselines
<p>background light grey: in bottom half of baselines and above 2.5%
<p>background white: in top half of baselines and below 97.5%
<p>background light violet-blue: in top 2.5% of baselines
<p>background cyan-blue: 0.0 to 0.5 LAS points above top baseline
<p>background yellow-green: 0.5 to 1.0 LAS points above top baseline
<p>background strong yellow: 1.0 to 2.0 LAS points above top baseline
<p>background pink: 2.0 to 3.0 LAS points above top baseline
<p>background strong red: 3.0 or more LAS points above top baseline
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
    method   = row[3]
    sample   = method[1]
    key = (language, parser, sample)
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
    row[1] = '%s\t%s' %(parser, sample)

rows.sort()
print('<p>%d rows</p>' %len(rows))

rows.append((None, 'n/a n/a'))

last_language = None
last_parser   = None
last_sample   = None
distribution  = None

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
s2text = {
    '-': 'bootstrap',
    'w': 'permutation',
    'x': '250%',
}
d2text = {
    '-': 'use all',
    'v': 'vanilla',
    'z': 'decaying',
}

class Distribution:

    def __init__(self, language, parser, sample):
        filename = None
        code = language + parser + sample
        for entry in os.listdir(os.environ['TT_DISTRIBUTIONS_DIR']):
            if not entry.startswith('distribution-') \
            or not entry.endswith('-dev.txt'):
                continue
            fields = entry.split('-')
            if len(fields) != 5 \
            or fields[1] != code \
            or fields[2] != '3':
                continue
            filename = entry
            break
        self.scores = []
        scores = self.scores
        if not filename:
            sys.stderr.write('Warning: no baseline distribution for %s.\n' %code)
        else:
            f = open('%s/%s' %(os.environ['TT_DISTRIBUTIONS_DIR'], filename), 'rb')
            while True:
                line = f.readline()
                if not line:
                    break
                scores.append(float(line.split()[0]))
            f.close()
        if self.scores:
            scores.sort()
            self.min_score = scores[0]
            self.max_score = scores[-1]
            num_scores = len(scores)
            if (num_scores % 2):
                # odd number of scores
                self.median = scores[(num_scores-1)/2]
            else:
                # even number of scores
                self.median = (scores[num_scores/2-1] + scores[num_scores/2])/2.0
            prune = int(0.025*len(scores))
            if prune:
                scores = self.scores[prune:-prune]
            self.score025 = scores[0]
            self.score975 = scores[-1]

    def colour(self, score):
        if not self.scores:
            return 'ffffff'
        colours = []
        for i in range(1001):
            offset = 0.0001 * i - 0.05
            colours.append(self.p_colour(score+offset))
        components = []
        for c_index in (0,1,2):
            value = 0.0
            for colour in colours:
                value += colour[c_index]
            components.append('%02x' %int(255.999 * value / float(len(colours))))
        return ''.join(components)

    def p_colour(self, score):
        if score < self.min_score:
            return (0.8, 0.2, 0.8)
        if score < self.score025:
            return (0.4, 0.4, 0.4)
        if score < self.median:
            return (0.8, 0.8, 0.8)
        if score < self.score975:
            return (1.0, 1.0, 1.0)
        if score < self.max_score:
            return (0.7, 0.6, 1.0)  # light violet-blue
        if score < self.max_score+0.5:
            return (0.6, 1.0, 1.0)  # light cyan-blue
        if score < self.max_score+1.0:
            return (0.8, 1.0, 0.6)  # light yellow-green
        if score < self.max_score+2.0:
            return (1.0, 1.0, 0.0)  # strong yellow
        if score < self.max_score+3.0:
            return (1.0, 0.7, 0.6)  # pink
        else:
            return (1.0, 0.0, 0.0)  # red

for row in rows:
    parser, sample = row[1].split()
    language = row[0]
    if (last_language, last_parser, last_sample) != (language, parser, sample) \
    and last_sample is not None:
        print('</table>')
        distribution = None
    if last_language != language and last_language is not None:
        print(legend)
    if language is None:
        break
    if last_language != language:
        print('<h2>%s</h2>' %l2text[language])
    if (last_language, last_parser) != (language, parser):
        print('<h3>%s</h3>' %p2text[parser])
    if (last_language, last_parser, last_sample) != (language, parser, sample):
        print('<h4>%s</h4>' %s2text[sample])
        print('<table cellpadding="4" border="1">')
        distribution = Distribution(language, parser, sample)
    print('<tr>')
    for i in range(0,baseline_column-5):
        print('<td>%s</td>' %row[i])
    # number of rounds
    print('<td>%s</td>' %row[baseline_column-2])
    # size in last round
    print('<td>%.1fk</td>' %(0.001*int(row[baseline_column-1])))
    bestbaselinescore = langandparser2bestbaseline[(language, parser, sample)]
    worstbaselinescore = langandparser2worstbaseline[(language, parser, sample)]
    text = row[baseline_column]
    score = float(text)
    tt_round = 0
    if distribution is None:
        tdcode = '<td>'
    else:
        tdcode = '<td bgcolor="#%s">' %distribution.colour(score)
    sccode = '<div title="Round %d with LAS %s">%.1f</div>' %(tt_round, text, score)
    if score == bestbaselinescore:
        print('%s<b>%s</b></td>' %(tdcode, sccode))
    else:
        print('%s%s</td>' %(tdcode, sccode))
    rowbaseline = score
    for text in row[baseline_column+1:]:
        tt_round += 1
        score = float(text)
        if distribution is None:
            tdcode = '<td>'
        else:
            tdcode = '<td bgcolor="#%s">' %distribution.colour(score)
        sccode = '<div title="Round %d with LAS %s">%.1f</div>' %(tt_round, text, score)
        if score >= bestbaselinescore:
            print('%s<b>%s</b></td>' %(tdcode, sccode))
        elif score < rowbaseline:
            print('%s<b><font color="red">%s</font></b></td>' %(tdcode, sccode))
        else:
            print('%s%s</td>' %(tdcode, sccode))
    last_sample = sample
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
