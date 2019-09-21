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

# optional args:
TEST_SET=$4
DEV_SET=$5

if [ -n "$TEST_SET" ]; then
    REAL_TEST_SET=$(realpath ${TEST_SET})
fi
if [ -n "$DEV_SET" ]; then
    REAL_DEV_SET=$(realpath ${DEV_SET})
fi

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

source ${PRJ_DIR}/config/locations.sh

PARSER_NAME=udpipe-future
PARSER_DIR=${UDPIPE_FUTURE_DIR}

if [ -n "$UDPIPE_FUTURE_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=${UDPIPE_FUTURE_LIB_PATH}:${LD_LIBRARY_PATH}
fi

if [ -n "$UDPIPE_FUTURE_CONDA" ]; then
    source ${CONDA_HOME}/etc/profile.d/conda.sh
    conda activate ${UDPIPE_FUTURE_CONDA}
fi

if [ -n "$UDPIPE_FUTURE_ENV" ]; then
    source ${UDPIPE_FUTURE_ENV}/bin/activate
fi

FAKE_TBID=xx_xxx

mkdir -p $MODELDIR

# The model is not complete without the conllu file as
# the checkpoint does not contain the vocabularies.

cp ${TRAIN_CONLLU} $MODELDIR/${FAKE_TBID}-ud-train.conllu
cd $MODELDIR

if [ -n "$TEST_SET" ]; then
    ln -s ${REAL_TEST_SET} ${FAKE_TBID}-ud-test.conllu
else
    # The parser complains if there is no test set.
    ln -s ${FAKE_TBID}-ud-train.conllu ${FAKE_TBID}-ud-test.conllu
fi

if [ -n "$DEV_SET" ]; then
    ln -s ${REAL_DEV_SET} ${FAKE_TBID}-ud-dev.conllu
fi

hostname > training.start

python ${PARSER_DIR}/ud_parser.py \
    --seed ${SEED}                \
    --logdir ./                   \
    ${FAKE_TBID}                  \
    2> stderr.txt     \
    > stdout.txt

#    --skip_incomplete_batches     \
#    --min_epoch_batches 3000      \
#    --epochs "4:1e-3,2:1e-4"      \


touch training.end

