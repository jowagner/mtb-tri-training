#!/bin/bash

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

test -z $1 && echo "Missing training conllu file"
test -z $1 && exit 1
TRAIN_CONLLU=$1

test -z $2 && echo "Missing seed"
test -z $2 && exit 1
SEED=$2

test -z $3 && echo "Missing language code"
test -z $3 && exit 1
LCODE=$3

test -z $4 && echo "Missing model output folder"
test -z $4 && exit 1
MODELDIR=$(realpath $4)

if [ -e "$MODELDIR" ]; then
    echo "Refusing to overwrite output folder"
    exit 1
fi

test -z $5 && echo "Missing batch size"
test -z $5 && exit 1
BATCH_SIZE=$5

test -z $6 && echo "Missing learning rate schedule"
test -z $6 && exit 1
LR_SCHEDULE=$6

MIN_EPOCH_SENTENCES=9600

# optional args:
EXTRA_OPTIONS="$7"
DEV_SET=$8
TEST_SET=$9

if [ -n "$DEV_SET" ]; then
    if [ -n "$TEST_SET" ]; then
        REAL_DEV_SET=$(realpath ${DEV_SET})
    else
        # only 1 monitoring set specified
        # --> make this the test set as udpipe-future
        #     requires a test set
        TEST_SET="$DEV_SET"
        unset DEV_SET
    fi
    REAL_TEST_SET=$(realpath ${TEST_SET})
fi

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

source ${PRJ_DIR}/config/locations.sh

NPZ=${FASTTEXT_NPZ_DIR}/fasttext-${LCODE}.npz

PARSER_NAME=udpipe-future
PARSER_DIR=${UDPIPE_FUTURE_DIR}

FINAL_MODELDIR=$MODELDIR
MODELDIR=${MODELDIR}-workdir

mkdir -p ${MODELDIR}

hostname > ${MODELDIR}/training.start

if [ -n "$UDPIPE_FUTURE_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=${UDPIPE_FUTURE_LIB_PATH}:${LD_LIBRARY_PATH}
fi

if [ -n "$UDPIPE_FUTURE_ICHEC_CONDA" ]; then
    module load conda/2
    source activate ${UDPIPE_FUTURE_ICHEC_CONDA}
fi

if [ -n "$UDPIPE_FUTURE_CONDA" ]; then
    source ${CONDA_HOME}/etc/profile.d/conda.sh
    conda activate ${UDPIPE_FUTURE_CONDA}
fi

if [ -n "$UDPIPE_FUTURE_ENV" ]; then
    source ${UDPIPE_FUTURE_ENV}/bin/activate
fi

FAKE_TBID=xx_xxx


# The model is not complete without the conllu file as
# the checkpoint does not contain the vocabularies.

if [ ${TRAIN_CONLLU: -4} == ".bz2" ]; then
    bzcat ${TRAIN_CONLLU} > $MODELDIR/${FAKE_TBID}-ud-train.conllu
else
    cp ${TRAIN_CONLLU} $MODELDIR/${FAKE_TBID}-ud-train.conllu
fi

cd $MODELDIR

if [ -e "${NPZ}" ]; then
    ln -s ${NPZ} fasttext.npz
else
    echo "Missing ${NPZ}"
    exit 1
fi

if [ -n "$TEST_SET" ]; then
    ln -s ${REAL_TEST_SET} ${FAKE_TBID}-ud-test.conllu
else
    # The parser complains if there is no test set.
    ln -s ${FAKE_TBID}-ud-train.conllu ${FAKE_TBID}-ud-test.conllu
fi

if [ -n "$DEV_SET" ]; then
    ln -s ${REAL_DEV_SET} ${FAKE_TBID}-ud-dev.conllu
fi

touch parser-training.start

MIN_EPOCH_BATCHES=$(expr ${MIN_EPOCH_SENTENCES} / ${BATCH_SIZE})
echo "Batch size is $BATCH_SIZE" >> training.start
echo "Minimum number of batches in each epoch: $MIN_EPOCH_BATCHES" >> training.start
if [ -e ${PARSER_DIR}/.git ] ; then
    echo "Parser commit:" $(git --git-dir=${PARSER_DIR}/.git describe --always) >> training.start
fi

python ${PARSER_DIR}/ud_parser.py \
    ${EXTRA_OPTIONS}              \
    --embeddings ${NPZ}           \
    --seed ${SEED}                \
    --logdir ./                   \
    --batch_size ${BATCH_SIZE}               \
    --min_epoch_batches ${MIN_EPOCH_BATCHES}  \
    --epochs "${LR_SCHEDULE}"     \
    ${FAKE_TBID}                  \
    2> stderr.txt     \
    > stdout.txt

#    --min_epoch_batches 3000      \   ## --> configure MIN_EPOCH_SENTENCES
#    --epochs "4:1e-3,2:1e-4"      \


touch training.end

cd /
if [ -e "$MODELDIR/checkpoint-inference-last.index" ]; then
    mv $MODELDIR $FINAL_MODELDIR
else
    SUFFIX=$(head -c 80 /dev/urandom | tr -dc 'A-Za-z0-9' | head -c 12)
    mv $MODELDIR ${FINAL_MODELDIR}-incomplete-${SUFFIX}
fi

