#!/bin/bash

for L in e h u v ; do
  for P in f h ; do
      sort -n < distr-baseline-tt-sim-${L}-${P}-best-of-12.txt > distr-baseline-tt-sim-${L}-${P}-best-of-12.txt.sorted
  done
  echo $L $(head -n 125001 \
  distr-baseline-tt-sim-${L}-f-best-of-12.txt.sorted | \
  tail -n 1) $(head -n 125001 \
  distr-baseline-tt-sim-${L}-h-best-of-12.txt.sorted | \
  tail -n 1)
done
