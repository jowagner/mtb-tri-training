#!/usr/bin/env python

import sys

las1 = float(sys.argv[1])
las2 = float(sys.argv[2])

err1 = 100.0 - las1
err2 = 100.0 - las2

print 'Error 1:', err1
print 'Error 2:', err2

reduction = err1 - err2

print 'Reduction:', reduction

percentage = 100.0 * reduction / err1

print '%.0f%%' %percentage


