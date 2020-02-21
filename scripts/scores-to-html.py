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

from collections import defaultdict
import os
import sys

print('<html><body>')

header = sys.stdin.readline().split()

langandparser2bestbaseline = {}
langandparser2worstbaseline = {}

baseline_column = header.index('0')
learner_column  = header.index('Learner')
testset_column  = header.index('TestSetIndex')

def unpack_score(score):
    if ':' in score:
        score, date, tokens, sentences = score.split(':')[:4]
    else:
        date, tokens, sentences = '????-??-??', '-1', '-1'
    return float(score), date, int(tokens), int(sentences)

def get_annotation_div(tt_round, text, date, n_tokens, n_sentences, score):
    return '&#10;'.join([
        '<div title="Round %d with LAS %s' %(
            tt_round, text.split(':')[0]
        ),
        'Model trained %s' %date,
        '%.1fk unlabelled tokens' %(n_tokens/1000.0),
        '%.1fk unlabelled sentences">%.1f</div>' %(
            n_sentences/1000.0, score
        ),
    ])

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
    # find best and worst baseline scores
    language = row[0]
    parser   = row[1]
    method   = row[3]
    sample   = method[1]
    key = (language, parser, sample)
    baselinescore, _, _, _ = unpack_score(row[baseline_column])
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
    # append sampling to parser information
    row[1] = '%s\t%s' %(parser, sample)

rows.sort()
print('<p>%d rows</p>' %len(rows))

rows.append((None, 'n/a n/a', None))

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
    '-': 'bootstrap samples of labelled data',
    'w': 'permutations of labelled data',
    'x': '250% of labelled data',
}
d2text = {
    '-': 'use all',
    'v': 'vanilla',
    'z': 'decaying',
}

class Distribution:

    def __init__(self, language, parser, sample, smooth = True):
        self.smooth = smooth
        filename = None
        if language is None:
            self.scores = True
            self.min_score =   0.0
            self.score025  =   2.5
            self.score250  =  25.0
            self.median    =  50.0
            self.score750  =  75.0
            self.score975  =  97.5
            self.max_score = 100.0
            return
        if sample == '-':
            sample = 's'
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
            backup = scores
            prune = int(0.025*len(scores))
            if prune:
                scores = self.scores[prune:-prune]
            self.score025 = scores[0]
            self.score975 = scores[-1]
            scores = backup
            prune = int(0.25*len(scores))
            if prune:
                scores = self.scores[prune:-prune]
            self.score250 = scores[0]
            self.score750 = scores[-1]

    def colour(self, score):
        if not self.scores:
            return 'ffffff'
        if self.smooth:
            indices = range(101)
        else:
            indices = [50,]
        colours = []
        for i in indices:
            for f in (0.000023, 0.000011):
                offset = f * (i - 50)
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
            return (0.00, 0.00, 0.00) # below 0.0: black
        if score < self.score025:
            return (0.60, 0.30, 0.15) #  0.0 -  2.5: brown
        if score < self.score250:
            return (0.67, 0.52, 0.45) #  2.5 - 25.0: brown-grey
        if score < self.median:
            return (0.75, 0.75, 0.75) # 25.0 - 50.0: grey
        if score < self.score750:
            return (1.00, 1.00, 1.00) # 50.0 - 75.0: white
        if score < self.score975:
            return (1.00, 1.00, 1.00) # 75.0 - 97.5: white
        if score < self.max_score:
            return (0.80, 0.60, 1.00) # 97.5 - 100: light violet-blue
        if score < self.max_score+0.4:
            return (0.65, 0.65, 1.00) # light blue
        if score < self.max_score+0.8:
            return (0.6, 1.0, 1.0)  # light cyan-blue
        if score < self.max_score+1.2:
            return (0.80, 1.0, 0.50)  # light yellow-green
        if score < self.max_score+1.8:
            return (1.0, 1.0, 0.0)  # strong yellow
        if score < self.max_score+2.4:
            return (1.0, 0.8, 0.5)  # orange-pink
        else:
            return (1.0, 0.5, 0.5)  # light red

legend = []
legend.append('<table>')
legend.append('<tr><td><b>black bold = above or matching best baseline for this language and parser</b></td></tr>')
legend.append('<tr><td><b><font color="red">red bold = below this seed and method\'s baseline</font></b></td></tr>')
distribution = Distribution(None, None, None, True)
for score, text in [
    ( -1.0, '<font color="red">background black: below worst baseline</font>'),
    (  0.0, 'blend'),
    (  0.1, 'background brown: in bottom 2.5% of baselines'),
    (  2.5, 'blend'),
    ( 10.0, 'background brown-grey: in bottom quartile but above 2.5%'),
    ( 25.0, 'blend'),
    ( 30.0, 'background grey: in 2nd quartile (25%-50%)'),
    ( 50.0, 'blend'),
    ( 73.8, 'background white: in upper half but below 97.5%'),
    ( 97.5, 'blend'),
    ( 99.0, 'background light violet-blue: in top 2.5% of baselines'),
    (100.0, 'blend'),
    (100.2, 'background light blue: 0.0 to 0.4 LAS points above top baseline'),
    (100.4, 'blend'),
    (100.6, 'background cyan-blue: 0.4 to 0.8 LAS points above top baseline'),
    (100.8, 'blend'),
    (101.0, 'background yellow-green: 0.8 to 1.2 LAS points above top baseline'),
    (101.2, 'blend'),
    (101.5, 'background strong yellow: 1.2 to 1.8 LAS points above top baseline'),
    (101.8, 'blend'),
    (102.1, 'background orange-pink: 1.8 to 2.4 LAS points above top baseline'),
    (102.4, 'blend'),
    (999.9, 'background pink-red: 2.4 or more LAS points above top baseline'),
]:
    legend.append('<tr><td bgcolor="#%s">%s</td></tr>' %(distribution.colour(score), text))
legend.append('</table>')
legend.append('<tr><td>(For scores at interval boundaries, the neighbouring colours are blended as shown above.)</td></tr>')
legend = '\n'.join(legend)

print(legend)

setting2rounds = defaultdict(lambda: 0)

def remove_zero(row):
    for i in range(len(row)):
        if row[i] == '0':
            row[i] = ''

def print_n_round_table(row_producer):
    print('<table cellpadding="4" border="1">')
    row_producer.print_top_header(left_columns = 1)
    row = []
    row.append('aug_size')
    row_producer.get_secondary_header(row)
    row = '</th><th>'.join(row)
    print('<tr><th>%s</th></tr>' %row)
    for p_augsize in '0123456789':
        augsize_k = int(0.5+5*(2.0**0.5)**int(p_augsize))
        row = []
        row.append('%s: %dk' %(p_augsize, augsize_k))
        row_producer.get_row(row, p_augsize)
        remove_zero(row)
        row = '</td><td align="right">'.join(row)
        print('<tr><td>%s</td></tr>' %row)
    print('</table>')

def print_n_round_for_language_and_parser(target_language, target_parser):
    class RowProducer:
        def __init__(self, target_language, target_parser):
            self.target_language = target_language
            self.target_parser   = target_parser
        def print_top_header(self, left_columns):
            print('<tr><th colspan="%d">' %left_columns)
            row = []
            row.append('')
            for p_sample in '-wx':
                row.append(s2text[p_sample])
            print('</th><th colspan="4">'.join(row))
            print('</th></tr>')
        def get_secondary_header(self, row):
            for p_sample in '-wx':
                for p_decay in '-v':
                    for p_oversampling in '-o':
                        p_code = '%s%s%s%s-%s' %(self.target_language, self.target_parser, p_oversampling, p_sample, p_decay)
                        row.append(p_code)
        def get_row(self, row, p_augsize):
            for p_sample in '-wx':
                for p_decay in '-v':
                    for p_oversampling in '-o':
                        setting_key = (self.target_language, p_augsize, self.target_parser, p_sample, p_decay, p_oversampling)
                        n_rounds = setting2rounds[setting_key]
                        row.append('%d' %n_rounds)
    print('<h4>Number of Rounds for Language %s and Parser %s</h4>' %(
        l2text[target_language], p2text[target_parser]
    ))
    row_producer = RowProducer(target_language, target_parser)
    print_n_round_table(row_producer)

def print_n_round_for_language_by_parser(target_language):
    class RowProducer:
        def __init__(self, target_language):
            self.target_language = target_language
        def print_top_header(self, left_columns):
            print('<tr><th colspan="%d">' %left_columns)
            row = []
            row.append('')
            for p_parser in 'fgh':
                row.append(p2text[p_parser])
            print('</th><th colspan="4">'.join(row))
            print('</th></tr>')
        def get_secondary_header(self, row):
            for p_parser in 'fgh':
                for p_decay in '-v':
                    for p_oversampling in '-o':
                        p_code = '%s%s%s%s-%s' %(self.target_language, p_parser, p_oversampling, '*', p_decay)
                        row.append(p_code)
        def get_row(self, row, p_augsize):
            for p_parser in 'fgh':
                for p_decay in '-v':
                    for p_oversampling in '-o':
                        values = []
                        for p_sample in '-wx':
                            setting_key = (self.target_language, p_augsize, p_parser, p_sample, p_decay, p_oversampling)
                            n_rounds = setting2rounds[setting_key]
                            values.append(n_rounds)
                        n_rounds = min(values)
                        row.append('%d' %n_rounds)
    print('<h4>Number of Rounds for Language %s by Parser</h4>' %(
        l2text[target_language]
    ))
    row_producer = RowProducer(target_language)
    print_n_round_table(row_producer)

def print_n_round_for_language_by_sample(target_language):
    class RowProducer:
        def __init__(self, target_language):
            self.target_language = target_language
        def print_top_header(self, left_columns):
            print('<tr><th colspan="%d">' %left_columns)
            row = []
            row.append('')
            for p_sample in '-wx':
                row.append(s2text[p_sample])
            print('</th><th colspan="4">'.join(row))
            print('</th></tr>')
        def get_secondary_header(self, row):
            for p_sample in '-wx':
                for p_decay in '-v':
                    for p_oversampling in '-o':
                        p_code = '%s%s%s%s-%s' %(self.target_language, '*', p_oversampling, p_sample, p_decay)
                        row.append(p_code)
        def get_row(self, row, p_augsize):
            for p_sample in '-wx':
                for p_decay in '-v':
                    for p_oversampling in '-o':
                        values = []
                        for p_parser in 'fgh':
                            setting_key = (self.target_language, p_augsize, p_parser, p_sample, p_decay, p_oversampling)
                            n_rounds = setting2rounds[setting_key]
                            values.append(n_rounds)
                        n_rounds = min(values)
                        row.append('%d' %n_rounds)
    print('<h4>Number of Rounds for Language %s by Seed Sampling Method</h4>' %(
        l2text[target_language]
    ))
    row_producer = RowProducer(target_language)
    print_n_round_table(row_producer)

def print_n_round_overall_by_parser():
    class RowProducer:
        def print_top_header(self, left_columns):
            print('<tr><th colspan="%d">' %left_columns)
            row = []
            row.append('')
            for p_parser in 'fgh':
                row.append(p2text[p_parser])
            print('</th><th colspan="4">'.join(row))
            print('</th></tr>')
        def get_secondary_header(self, row):
            for p_parser in 'fgh':
                for p_decay in '-v':
                    for p_oversampling in '-o':
                        p_code = '%s%s%s%s-%s' %('*', p_parser, p_oversampling, '*', p_decay)
                        row.append(p_code)
        def get_row(self, row, p_augsize):
            for p_parser in 'fgh':
                for p_decay in '-v':
                    for p_oversampling in '-o':
                        values = []
                        for p_sample in '-wx':
                            for p_language in 'ehuv':
                                setting_key = (p_language, p_augsize, p_parser, p_sample, p_decay, p_oversampling)
                                n_rounds = setting2rounds[setting_key]
                                values.append(n_rounds)
                        n_rounds = min(values)
                        row.append('%d' %n_rounds)
    print('<h4>Number of Rounds overall by Parser</h4>')
    row_producer = RowProducer()
    print_n_round_table(row_producer)

def print_n_round_overall_by_sample():
    class RowProducer:
        def print_top_header(self, left_columns):
            print('<tr><th colspan="%d">' %left_columns)
            row = []
            row.append('')
            for p_sample in '-wx':
                row.append(s2text[p_sample])
            print('</th><th colspan="4">'.join(row))
            print('</th></tr>')
        def get_secondary_header(self, row):
            for p_sample in '-wx':
                for p_decay in '-v':
                    for p_oversampling in '-o':
                        p_code = '%s%s%s%s-%s' %('*', '*', p_oversampling, p_sample, p_decay)
                        row.append(p_code)
        def get_row(self, row, p_augsize):
            for p_sample in '-wx':
                for p_decay in '-v':
                    for p_oversampling in '-o':
                        values = []
                        for p_parser in 'fgh':
                            for p_language in 'ehuv':
                                setting_key = (p_language, p_augsize, p_parser, p_sample, p_decay, p_oversampling)
                                n_rounds = setting2rounds[setting_key]
                                values.append(n_rounds)
                        n_rounds = min(values)
                        row.append('%d' %n_rounds)
    print('<h4>Number of Rounds overall by Seed Sampling Method</h4>')
    row_producer = RowProducer()
    print_n_round_table(row_producer)

def print_n_round_for_language(target_language):
    print_n_round_for_language_by_parser(target_language)
    print_n_round_for_language_by_sample(target_language)

def print_n_round_overall():
    print_n_round_overall_by_parser()
    print_n_round_overall_by_sample()

for row in rows:
    parser, sample = row[1].split()
    language = row[0]
    augsize = row[2]
    if (last_language, last_parser, last_sample) != (language, parser, sample) \
    and last_sample is not None:
        print('</table>')
        if (last_language, last_parser) != (language, parser) \
        and last_language is not None \
        and last_parser is not None:
            print_n_round_for_language_and_parser(last_language, last_parser)
        distribution = None
    if last_language != language and last_language is not None:
        print(legend)
        print_n_round_for_language(last_language)
    if language is None:
        break
    if last_language != language:
        print('<h2>%s</h2>' %l2text[language])
    if (last_language, last_parser) != (language, parser):
        print('<h3>%s</h3>' %p2text[parser])
    if (last_language, last_parser, last_sample) != (language, parser, sample):
        print('<h4>Using %s for round 0 models</h4>' %s2text[sample])
        print('<table cellpadding="4" border="1">')
        distribution = Distribution(language, parser, sample)
        last_augsize = None
    if last_augsize is not None and last_augsize != augsize:
        print('<tr><td></td></tr>')
    print('<tr>')
    for i in range(0,baseline_column-5):
        if i == baseline_column-7:
            augsize_k = int(0.5+5*(2.0**0.5)**int(row[i]))
            print('<td>%s: %dk</td>' %(row[i], augsize_k))
            continue
        print('<td>%s</td>' %row[i])
    # number of rounds
    n_rounds = row[baseline_column-2]
    print('<td>%s</td>' %n_rounds)
    oversampling, _, agreement, decay = row[baseline_column-6]
    if agreement != '-':
        raise ValueError('Only default agreement supported so far')
    setting_key = (language, augsize, parser, sample, decay, oversampling)
    setting2rounds[setting_key] = int(n_rounds)
    # size in last round
    print('<td>%.1fk</td>' %(0.001*int(row[baseline_column-1])))
    bestbaselinescore = langandparser2bestbaseline[(language, parser, sample)]
    worstbaselinescore = langandparser2worstbaseline[(language, parser, sample)]
    text = row[baseline_column]
    score, date, n_tokens, n_sentences = unpack_score(text)
    tt_round = 0
    if distribution is None:
        tdcode = '<td align="right">'
    else:
        tdcode = '<td bgcolor="#%s" align="right">' %distribution.colour(score)
    sccode = get_annotation_div(tt_round, text, date, n_tokens, n_sentences, score)
    if score == bestbaselinescore:
        print('%s<b>%s</b></td>' %(tdcode, sccode))
    else:
        print('%s%s</td>' %(tdcode, sccode))
    rowbaseline = score
    for text in row[baseline_column+1:]:
        tt_round += 1
        score, date, n_tokens, n_sentences = unpack_score(text)
        if distribution is None:
            tdcode = '<td align="right">'
        else:
            tdcode = '<td bgcolor="#%s" align="right">' %distribution.colour(score)
        sccode = get_annotation_div(tt_round, text, date, n_tokens, n_sentences, score)
        if score >= bestbaselinescore:
            print('%s<b>%s</b></td>' %(tdcode, sccode))
        elif score < rowbaseline:
            print('%s<b><font color="red">%s</font></b></td>' %(tdcode, sccode))
        else:
            print('%s%s</td>' %(tdcode, sccode))
    last_sample = sample
    last_parser = parser
    last_language = language
    last_augsize = augsize

print('<body></html>')

print('<h2>Overall Number of Rounds</h2>')
print_n_round_overall()

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
