#!/bin/bash

for L in e h u v ; do
  for P in f h ; do
      echo "== $L $P =="
      SCORE=$(head -n 1 tt-${L}-${P}-best-of-12.txt)
      tail -n +2 tt-${L}-${P}-best-of-12.txt | fgrep --color=always $SCORE
      echo
  done
done
