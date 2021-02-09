#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import os
import sys

from distribution import Distribution

target_parsers = 'fh'
target_oversmpls  = '-o'
target_min_rounds = 4   # do not include a run if it has fewer rounds

if len(target_parsers) < 2:
    target_parsers = target_parsers + target_parsers[0]

assert len(target_oversmpls) >= 2

header = sys.stdin.readline().split()

baseline_column = header.index('0')
language_column = header.index('Language')
parser_column   = header.index('Parser')
try:
    run_column  = header.index('Run')
except:
    run_column  = None
aug_column      = header.index('AugmentSizeIndex')
method_column   = header.index('Method')
learner_column  = header.index('Learner')
testset_column  = header.index('TestSetIndex')
rounds_column   = header.index('Rounds')

example_header = """Language        Parser  Run     AugmentSizeIndex        Method  NumberOfLearners        Learner TestSetIndex    Rounds  UnlabelledTokensInLastRound    0       1
"""

def unpack_score(score):
    if ':' in score:
        score, date, tokens, sentences = score.split(':')[:4]
    else:
        date, tokens, sentences = '????-??-??', '-1', '-1'
    return float(score), date, int(tokens), int(sentences)

graphs = {}
languages = set()
parsers = set()
sample_types = set()
oversmpl_types = set()

while True:
    line = sys.stdin.readline()
    if not line:
        break
    row = line.split()
    if row[learner_column] != 'Ensemble':
        continue
    if row[testset_column] != '0':
        continue
    # find best and worst baseline scores
    language = row[language_column]
    parser   = row[parser_column]
    if parser == 't':
        parser = 'f'
    if parser not in target_parsers:
        continue
    method   = row[method_column]
    oversampling, sample, agreement, decay = method
    try:
        augsize = row[aug_column]
    except:
        raise ValueError('No index %d in row %r' %(aug_column, row))
    if run_column is None:
        run = '1'
    else:
        run = row[run_column]
    n_rounds = int(row[rounds_column])
    if n_rounds < target_min_rounds:
        continue
    languages.add(language)
    parsers.add(parser)
    sample_types.add(sample)
    oversmpl_types.add(oversampling)
    graph_key = (language, parser, oversampling)
    setting_key = (sample, agreement, decay, augsize)
    if not graph_key in graphs:
        graphs[graph_key] = {}
    graph = graphs[graph_key]
    if not setting_key in graph:
        graph[setting_key] = []
    setting = graph[setting_key]
    experiment_code = '%s%s%s%s%s%s%d%s' %(
        language, parser, oversampling, sample,
        agreement, decay,
        2 + int(run),
        augsize
    )
    setting.append((-n_rounds, run, experiment_code, row))

summary = []
for lang_index, language in enumerate(sorted(list(languages))):
    sys.stderr.write('\n\n== Language %s ==\n\n' %language)
    eligible_setting_keys = None
    for parser in target_parsers:
        for oversmpl in target_oversmpls:
            graph_key = (language, parser, oversmpl)
            if not graph_key in graphs:
                setting_keys_for_graph_key = set()
            else:
                setting_keys_for_graph_key = set(graphs[graph_key].keys())
            if eligible_setting_keys is None:
                eligible_setting_keys = setting_keys_for_graph_key
            else:
                # intersection
                eligible_setting_keys = eligible_setting_keys.intersection(setting_keys_for_graph_key)
    if not eligible_setting_keys:
        continue
    sys.stderr.write('Usable settings:\n')
    sk2min_rounds = {}
    for setting_key in sorted(list(eligible_setting_keys)):
        min_rounds = []
        for run_index in (0,1):
            rounds = []
            for parser in target_parsers:
                for oversmpl in target_oversmpls:
                    graph_key = (language, parser, oversmpl)
                    setting = graphs[graph_key][setting_key]
                    if not run_index:
                        setting.sort()
                    if len(setting) > run_index:
                        neg_n_rounds, run, exp_code, row = setting[run_index]
                        rounds.append(-neg_n_rounds)
            if len(rounds) == len(target_parsers) * len(target_oversmpls):
                assert len(min_rounds) == run_index 
                min_rounds.append(min(rounds))
        #setting = ( # (-int(n_rounds), run, experiment_code, row)
        sys.stderr.write('\t___%s%s%s_%s\t%r\n' %(tuple(setting_key) + (min_rounds,)))
        sk2min_rounds[setting_key] = min_rounds
    is_first_parser = True
    for parser in target_parsers:
        # get list of columns for header
        exp_codes = []
        for oversmpl in sorted(list(oversmpl_types)):
            graph_key = (language, parser, oversmpl)
            if not graph_key in graphs:
                assert oversmpl not in target_oversmpls
                continue
            graph = graphs[graph_key]
            for setting_key in sorted(list(eligible_setting_keys)):
                if not setting_key in graph:
                    assert oversmpl not in target_oversmpls
                    continue
                setting = graph[setting_key]
                min_rounds = sk2min_rounds[setting_key]
                for run_index, rounds in enumerate(min_rounds):
                    if oversmpl not in target_oversmpls:
                        setting.sort()
                    if len(setting) > run_index:
                        neg_n_rounds, run, exp_code, row = setting[run_index]
                        exp_codes.append(exp_code)
        out = open('out_%s_%s.tsv' %(language, parser), 'wb')
        # write header
        header = []
        header.append('X')
        for column in 'Average Min P025 P250 Median P750 P975 Max'.split():
            header.append(column)
        for exp_code in exp_codes:
            header.append(exp_code)
        out.write('%s\n' %('\t'.join(header)))
        # prepare table contents
        sys.stderr.write('\nParser %s:\n' %parser)
        x_base = 1.0
        characteristic = []
        for oversmpl in sorted(list(oversmpl_types)):
            graph_key = (language, parser, oversmpl)
            if not graph_key in graphs:
                assert oversmpl not in target_oversmpls
                continue
            graph = graphs[graph_key]
            max_rounds = 0
            best_scores = []
            for setting_key in sorted(list(eligible_setting_keys)):
                if not setting_key in graph:
                    assert oversmpl not in target_oversmpls
                    continue
                setting = graph[setting_key]
                min_rounds = sk2min_rounds[setting_key]
                for run_index, rounds in enumerate(min_rounds):
                    if rounds > max_rounds:
                        max_rounds = rounds
                    if len(setting) > run_index:
                        neg_n_rounds, run, exp_code, row = setting[run_index]
                        column = exp_codes.index(exp_code)
                        best_score = 0.0
                        scores_tested = 0
                        for t, info in enumerate(row[baseline_column:]):
                            if t > rounds:
                                continue
                            score, _, _, _ = unpack_score(info)
                            x = x_base + t
                            output_row = []
                            output_row.append('%.2f' %x)
                            for _ in range(column+8):
                                output_row.append('')
                            output_row.append('%.9f' %score)
                            out.write('%s\n' %('\t'.join(output_row)))
                            if score > best_score:
                                best_score = score
                            scores_tested = scores_tested + 1
                        best_scores.append(best_score)
                        sys.stderr.write('\t___%s%s%s_%s\t%s\t%d\t%.9f\t(best of %d scores)\n' %(tuple(setting_key) + (oversmpl, run_index, best_score, scores_tested)))
            if best_scores:
                average_score = sum(best_scores) / float(len(best_scores))
                sys.stderr.write('\taverage best score for %s: %.9f (average of %d scores)\n' %(oversmpl, average_score, len(best_scores)))
            if oversmpl in target_oversmpls:
                assert len(best_scores) > 0
                average_score = sum(best_scores) / float(len(best_scores))
                characteristic.append(average_score)
            x_base = x_base + max_rounds + 2
        out.close()
        assert len(characteristic) >= 2
        summary_row = []
        if is_first_parser:
            summary_row.append('\\multirow{2}{*}{%s}' %language)
        else:
            summary_row.append('')
        summary_row.append(parser)
        best_score = max(characteristic)
        for score in characteristic:
            if score == best_score:
                summary_row.append('\\textbf{%.1f}' %score)
            else:
                summary_row.append('%.1f' %score)
        if len(target_oversmpls) == 2:
            summary_row.append('%.3f' %(characteristic[1]-characteristic[0]))
        summary.append(' & '.join(summary_row))
        is_first_parser = False

header = []
header.append('\\textbf{Language}')
header.append('\\textbf{Parser}')
for oversmpl in sorted(list(set(target_oversmpls) & oversmpl_types)):
    header.append('\\textbf{%s}' %oversmpl)
if len(target_oversmpls) == 2:
    header.append('\\textbf{$\Delta$}')
out = open('summary.tex', 'wb')
out.write(' & '.join(header))
out.write('\\\\\n\\hline\n')
for row in summary:
    out.write(row+' \\\\\n')
out.close()
