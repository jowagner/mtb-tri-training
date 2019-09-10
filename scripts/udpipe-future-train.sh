#!/bin/bash

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

test -z $1 && echo "Missing training conllu file"
test -z $1 && exit 1
TRAIN_CONLLU=$1

test -z $2 && echo "Missing seed"
test -z $2 && exit 1
SEED=$2

test -z $3 && echo "Missing model output folder"
test -z $3 && exit 1
MODELDIR=$3

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

source ${PRJ_DIR}/config/locations.sh

PARSER_NAME=udpipe-future
PARSER_DIR=${UDPIPE_FUTURE_DIR}

if [ -n "$UDPIPE_FUTURE_ENV" ]; then
    source ${UDPIPE_FUTURE_ENV}/bin/activate
fi

FAKE_TBID=xx_xxx

mkdir -p $MODELDIR

# The model is not complete without the conllu file as
# the checkpoint does not contain the vocabularies.

cp ${TRAIN_CONLLU} $MODELDIR/${FAKE_TBID}-ud-train.conllu
cd $MODELDIR

# The parser complains if there is no test set.

ln -s ${FAKE_TBID}-ud-train.conllu ${FAKE_TBID}-ud-test.conllu

hostname > training.start

python ${PARSER_DIR}/ud_parser.py \
    --seed ${SEED}                \
    --skip_incomplete_batches     \
    --min_epoch_batches 36        \
    --epochs "12:1e-3,5:1e-4"     \
    --logdir ./                   \
    ${FAKE_TBID}

touch training.end

