#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import os
import sys

class Distribution:

    def __init__(self, language, parser, sample, smooth = True, with_info = False):
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
        if language == 't':
            parser = 't'
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
        if with_info:
            self.info = []
            info = self.info
        if not filename:
            sys.stderr.write('Warning: no baseline distribution for %s.\n' %code)
        else:
            f = open('%s/%s' %(os.environ['TT_DISTRIBUTIONS_DIR'], filename), 'rb')
            while True:
                line = f.readline()
                if not line:
                    break
                scores.append(float(line.split()[0]))
                if with_info:
                    info.append(line.rstrip().split('\t'))
            f.close()
        if self.scores:
            self.average = sum(scores) / float(len(scores))
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
        steps = 100
        assert (steps % 2) == 0
        if self.smooth in (True, 'at boundary', 1):
            # smooth colours when very close to boundary between two colours
            indices = range(steps+1)
            radius1 = 0.00115 / float(steps/2)
            radius2 = 0.00055 / float(steps/2)
        elif self.smooth in (False, 'none', None, 0):
            # do not blend colours
            indices = [steps//2,]
            radius1 = 0.0
            radius2 = 0.0
        elif self.smooth in ('extra smooth', 2):
            # very smooth blends (not matching legend)
            indices = range(steps+1)
            radius1 = 0.30050 / float(steps/2)
            radius2 = 0.05450 / float(steps/2)
        colours = []
        for i in indices:
            for f in (radius1, radius2):
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
            #return (0.80, 0.60, 1.00) # 97.5 - 100: light violet-blue
            return (0.65, 0.65, 1.00) # 97.5 - 100: light blue
        #if score < self.max_score+0.4:
            #return (0.65, 0.65, 1.00) # light blue
        if score < self.max_score+0.6:
            return (0.6, 1.0, 1.0)  # light cyan-blue
        if score < self.max_score+1.2:
            return (0.80, 1.0, 0.50)  # light yellow-green
        if score < self.max_score+1.8:
            return (1.0, 1.0, 0.0)  # strong yellow
        if score < self.max_score+2.4:
            return (1.0, 0.8, 0.5)  # orange-pink
        if score < self.max_score+3.0:
            return (1.0, 0.5, 0.5)  # light red
        else:
            return (1.0, 0.5, 1.0)  # light magenta-purple

def main():
    language = sys.argv[1]
    parser   = sys.argv[2]
    x = 1.0
    for sample in '-wxt':
        distr = Distribution(language, parser, sample)
        if not distr.scores:
            continue
        sys.stdout.write('# sampling: %s\n' %sample)
        sys.stdout.write('%.1f\t%.9f\t# min\n'     %(x, distr.min_score))
        sys.stdout.write('%.1f\t%.9f\t# 2.5%%\n'   %(x+0.5, distr.score025))
        sys.stdout.write('%.1f\t%.9f\t# 25%%\n'    %(x+1.0, distr.score250))
        sys.stdout.write('%.1f\t%.9f\t# median\n'  %(x+2.0, distr.median))
        sys.stdout.write('%.1f\t%.9f\t# 75%%\n'    %(x+3.0, distr.score750))
        sys.stdout.write('%.1f\t%.9f\t# 97.5%%\n'  %(x+3.5, distr.score975))
        sys.stdout.write('%.1f\t%.9f\t# max\n'     %(x+4.0, distr.max_score))
        sys.stdout.write('%.1f\t%.9f\t# average\n' %(x+4.5, distr.average))
        x += 10.0

if __name__ == "__main__":
    main()

