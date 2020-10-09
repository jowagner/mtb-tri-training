#!/bin/bash
  
D=$1

source $HOME/tri-training/mtb-tri-training/config/locations.sh

cd $HOME/tri-training/mtb-tri-training/workdirs

echo $(hostname) $(date) >> distr-$D.start

find | $HOME/tri-training/mtb-tri-training/scripts/get-baseline-distribution.py --distribution $D \
    2> distr-${D}-stderr.txt  \
    >  distr-${D}-stdout.txt

touch distr-$D.end

