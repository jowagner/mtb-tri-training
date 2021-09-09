#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

from collections import defaultdict
import os
import random
import sys

from distribution import Distribution

target_parsers = 'fghi'
target_min_rounds = 4   # do not include a run if it has fewer rounds

target_samples = 'x'

bin_width = 0.2

baseline_distr_samples = 2500000   # reduce to small number when not iterested in baseline distribution or debugging

if len(target_parsers) < 2:
    target_parsers = target_parsers + target_parsers[0]

header = sys.stdin.readline().split()

max_runs_per_setting = 5

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
    if n_rounds < target_min_rounds \
    or augsize not in '68A'    \
    or sample not in target_samples  \
    or agreement != '-'  \
    or decay not in '-ovz':
        continue
    languages.add(language)
    parsers.add(parser)
    graph_key = (language, parser)
    setting_key = (sample, oversampling, agreement, decay, augsize)
    if decay == 'o':
        # treat d=0.71 as d=1
        setting_key = (sample, oversampling, agreement, '-', augsize)
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

def update_table(table, key, column_index, value):
    if not key in table:
        table[key] = []
    row = table[key]
    while len(row) <= column_index:
        row.append(None)
    row[column_index] = value

def write_table(out, table):
    for key in sorted(list(table.keys())):
        row = []
        row.append('%.9f' %key)
        for value in table[key]:
            if value is None:
                row.append('')
            else:
                row.append('%.9f' %value)
        out.write('\t'.join(row))
        out.write('\n')

def write_freq_distr(table, bin2freq, bin_width, column_index = 0):
    total = float(sum(bin2freq.values()))
    bin_keys = sorted(list(bin2freq.keys()))
    last_key = None
    max_gap = 1.375 * bin_width
    for key in bin_keys:
        # detect skipped keys for which we want to write 0
        if last_key is not None:
            gap = key - last_key
            if gap > max_gap:
                # number of steps
                steps = int(0.5+gap/bin_width)
                for step in range(1,steps):
                    score = last_key+step*bin_width
                    s_key = bin_width * round(score / bin_width)
                    update_table(table, s_key, column_index, 0.0)
        freq = bin2freq[key]
        update_table(table, key, column_index, freq/total)
        last_key = key

summary = []
for lang_index, language in enumerate(sorted(list(languages))):
    sys.stderr.write('\n\n== Language %s ==\n\n' %language)
    eligible_setting_keys = None
    for parser in target_parsers:
        if True:
            graph_key = (language, parser)
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
    out = open('settings_%s.tsv' %language, 'wb')
    header = 'SettingKey Index Parser ExpCode UseRounds TotalRounds Run'.split()
    out.write('\t'.join(header))
    out.write('\n')
    sk2min_rounds = {}
    for setting_key in sorted(list(eligible_setting_keys)):
        min_rounds = []
        for run_index in range(max_runs_per_setting):
            rounds = []
            for parser in target_parsers:
                if True:
                    graph_key = (language, parser)
                    setting = graphs[graph_key][setting_key]
                    if not run_index:   # first time:
                        setting.sort()  # order by n_rounds (descending)
                    try:
                        neg_n_rounds, run, exp_code, row = setting[run_index]
                        rounds.append(-neg_n_rounds)
                    except IndexError:
                        pass
            if len(rounds) == len(target_parsers):
                assert len(min_rounds) == run_index 
                min_rounds.append(min(rounds))
                for parser in target_parsers:
                    graph_key = (language, parser)
                    setting = graphs[graph_key][setting_key]
                    neg_n_rounds, run, exp_code, row = setting[run_index]
                    row = []
                    row.append(' '.join((setting_key)))
                    row.append('%d' %run_index)
                    row.append(parser)
                    row.append(exp_code)
                    row.append('%d' %(min(rounds)))
                    row.append('%d' %(-neg_n_rounds))
                    row.append(run)
                    out.write('\t'.join(row))
                    out.write('\n')
            else:
                # no run at this run_index for at least one parser
                # --> no point in trying next run_index
                break
        #setting = ( # (-int(n_rounds), run, experiment_code, row)
        sys.stderr.write('\t__%s%s%s%s_%s\t%r\n' %(tuple(setting_key) + (min_rounds,)))
        sk2min_rounds[setting_key] = min_rounds
    out.close()
    is_first_parser = True
    for parser in target_parsers:
        # get list of columns for header
        exp_codes = []
        if True:
            graph_key = (language, parser)
            graph = graphs[graph_key]
            for setting_key in sorted(list(eligible_setting_keys)):
                setting = graph[setting_key]
                setting.sort()
                min_rounds = sk2min_rounds[setting_key]
                for run_index, rounds in enumerate(min_rounds):
                    if len(setting) > run_index:
                        neg_n_rounds, run, exp_code, row = setting[run_index]
                        exp_codes.append(exp_code)
        out = open('out_%s_%s.tsv' %(language, parser), 'wb')
        table = {}
        # write header
        header = []
        for column in 'LAS Baseline Picking Tri-training'.split():
            header.append(column)
        out.write('%s\n' %('\t'.join(header)))
        bin2freq = defaultdict(lambda: 0)
        for sample in target_samples:
            distr = Distribution(language, parser, sample)
            for score in distr.scores:
                bin_key = bin_width * round(score / bin_width)
                bin2freq[bin_key] += 1
        write_freq_distr(table, bin2freq, bin_width, column_index = 0)
        # prepare table contents
        sys.stderr.write('\nParser %s:\n' %parser)
        x_base = 1.0
        characteristic = []
        bin2freq = defaultdict(lambda: 0)
        picks = []
        if True:
            bmout = open('best_model_%s_%s.tsv' %(language, parser), 'wb')
            header = 'SettingKey Index Parser ExpCode BaselineLAS BestRound BestLAS'.split()
            bmout.write('\t'.join(header))
            bmout.write('\n')
            graph_key = (language, parser)
            graph = graphs[graph_key]
            max_rounds = 0
            best_scores = []
            details = []
            for setting_key in sorted(list(eligible_setting_keys)):
                setting = graph[setting_key]
                sample  = setting_key[0]
                min_rounds = sk2min_rounds[setting_key]
                for run_index, rounds in enumerate(min_rounds):
                    bmrow = []
                    bmrow.append(' '.join(setting_key))
                    bmrow.append('%d' %run_index)
                    bmrow.append(parser)
                    if rounds > max_rounds:
                        max_rounds = rounds
                    if len(setting) > run_index:
                        neg_n_rounds, run, exp_code, row = setting[run_index]
                        bmrow.append(exp_code)
                        bmrow.append('%.9f' %(unpack_score(row[baseline_column])[0]))
                        column = exp_codes.index(exp_code)
                        best_score = 0.0
                        best_round = -1
                        scores_tested = 0
                        for t, info in enumerate(row[baseline_column:]):
                            if t > rounds:
                                continue
                            score, _, _, _ = unpack_score(info)
                            x = x_base + t
                            if score > best_score:
                                best_score = score
                                best_round = t
                            scores_tested = scores_tested + 1
                        bmrow.append('%02d' %best_round)
                        bmrow.append('%.9f' %best_score)
                        best_scores.append(best_score)
                        details.append((setting_key, run_index, rounds, max_rounds, run, exp_code, row))
                        sys.stderr.write('\t__%s%s%s%s_%s\t*\t%d\t%.9f\t(best of %d scores)\n' %(tuple(setting_key) + (run_index, best_score, scores_tested)))
                        picks.append((scores_tested, sample))
                    bmout.write('\t'.join(bmrow))
                    bmout.write('\n')
            bmout.close()
            if best_scores:
                average_score = sum(best_scores) / float(len(best_scores))
                sys.stderr.write('\taverage best score: %.9f (average of %d scores)\n' %(average_score, len(best_scores)))
                for score in best_scores:
                    bin_key = bin_width * round(score / bin_width)
                    bin2freq[bin_key] += 1
                out_b = open('tt-%s-%s-best-of-%d.txt' %(
                    language, parser, len(best_scores)
                ), 'w')
                max_score = max(best_scores)
                out_b.write('%.9f\n' %max_score)
                for info_idx, info in enumerate(details):
                    if best_scores[info_idx] >= max_score:
                        out_b.write('[%d]\t%r\n' %(info_idx, info))
                out_b.close()
            if True:
                assert len(best_scores) > 0
                average_score = sum(best_scores) / float(len(best_scores))
                characteristic.append(average_score)
            x_base = x_base + max_rounds + 2
        write_freq_distr(table, bin2freq, bin_width, column_index = 2)
        # simulate what distribution of scores we would have gotten
        # if each tri-training result was just a random baseline ensemble
        bin2freq = defaultdict(lambda: 0)
        lps2distr = {}
        for _, sample in picks:
            if (language, parser, sample) not in lps2distr:
                distr = Distribution(language, parser, sample)
                lps2distr[(language, parser, sample)] = distr
        out_a = open('distr-baseline-tt-sim-%s-%s-all.txt' %(language, parser), 'w')
        out_b = open('distr-baseline-tt-sim-%s-%s-best-of-12.txt' %(language, parser), 'w')
        for _ in range(baseline_distr_samples):
            best_score = None
            for pick, sample in picks:
                distr = lps2distr[(language, parser, sample)]
                scores = random.sample(distr.scores, pick)
                score = max(scores)
                out_a.write('%.9f\n' %score)
                if best_score is None or score > best_score:
                    best_score = score
                bin_key = bin_width * round(score / bin_width)
                bin2freq[bin_key] += 1
            out_b.write('%.9f\n' %best_score)
        out_a.close()
        out_b.close()
        write_freq_distr(table, bin2freq, bin_width, column_index = 1)
        write_table(out, table)
        out.close()
        assert len(characteristic) == 1
        summary_row = []
        if is_first_parser:
            summary_row.append('\\multirow{2}{*}{%s}' %language)
        else:
            summary_row.append('                  ')
        summary_row.append(parser)
        best_score = max(characteristic)
        for score in characteristic:
            if score == best_score:
                summary_row.append('\\textbf{%.1f}' %score)
            else:
                summary_row.append('        %.1f ' %score)
        summary.append(' & '.join(summary_row))
        is_first_parser = False

header = []
header.append('\\textbf{Language}')
header.append('\\textbf{Parser}')
header.append('\\textbf{LAS}')
out = open('summary.tex', 'wb')
out.write(' & '.join(header))
out.write('\\\\\n\\hline\n')
for row in summary:
    out.write(row+' \\\\\n')
out.close()
