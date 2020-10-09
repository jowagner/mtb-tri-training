#!/bin/bash
  
D=$1
P=$2

source $HOME/tri-training/mtb-tri-training/config/locations.sh

cd $HOME/tri-training/mtb-tri-training/workdirs

echo $(hostname) $(date) >> distr-${D}-${P}.start

find | $HOME/tri-training/mtb-tri-training/scripts/get-baseline-distribution.py --distribution $D --part $P \
    2> distr-${D}-${P}-stderr.txt  \
    >  distr-${D}-${P}-stdout.txt

touch distr-${D}-${P}.end

