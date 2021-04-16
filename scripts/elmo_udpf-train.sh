#!/bin/bash

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

test -z $1 && echo "Missing training conllu file"
test -z $1 && exit 1
TRAIN_CONLLU=$1

test -z $2 && echo "Missing training npz file"
test -z $2 && exit 1
TRAIN_NPZ=$2


test -z $3 && echo "Missing seed"
test -z $3 && exit 1
SEED=$3

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

test -z $7 && echo "Missing language code"
test -z $7 && exit 1
LANG_CODE=$7

test -z $8 && echo "Missing embedding code"
test -z $8 && exit 1
EMB_CODE=$8

# optional args:
EXTRA_OPTIONS="$9"
DEV_SET=${10}
DEV_NPZ=${11}
TEST_SET=${12}
TEST_NPZ=${13}

if [ -e "$TRAIN_NPZ" ]; then
    echo "training npz ready" > /dev/null
else
    echo "Waiting for training npz"
    sleep 60
fi

if [ -n "$DEV_SET" ]; then
    if [ -e "$DEV_NPZ" ]; then
        echo "1st monitoring npz ready" > /dev/null
    else
        echo "Waiting for 1st monitoring npz"
        sleep 60
    fi
    if [ -n "$TEST_SET" ]; then
        if [ -e "$TEST_NPZ" ]; then
            echo "2nd monitoring npz ready" > /dev/null
        else
            echo "Waiting for 2nd monitoring npz"
            sleep 60
        fi
        REAL_DEV_SET=$(realpath ${DEV_SET})
        N_MONITORS=2
    else
        # only 1 monitoring set specified
        # --> make this the test set as udpipe-future
        #     requires a test set
        TEST_SET="$DEV_SET"
        TEST_NPZ="$DEV_NPZ"
        unset DEV_SET
        N_MONITORS=1
    fi
    REAL_TEST_SET=$(realpath ${TEST_SET})
else
    N_MONITORS=0
fi


test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

FINAL_MODELDIR=$MODELDIR
MODELDIR=${MODELDIR}-workdir

test -e ${MODELDIR} && echo "Conflicting pre-existing workdir ${MODELDIR}"
test -e ${MODELDIR} && exit 1
#if [ -e ${FINAL_MODELDIR} ]; then
#    mv ${FINAL_MODELDIR} ${MODELDIR}
#fi
mkdir -p ${MODELDIR}

hostname > ${MODELDIR}/training.start

ELMO_FILE_PREFIX=elmo

ln -s ${TRAIN_NPZ} ${MODELDIR}/${ELMO_FILE_PREFIX}-train.npz

if [ -n "$TEST_SET" ]; then
    ln -s ${TEST_NPZ} ${MODELDIR}/${ELMO_FILE_PREFIX}-test.npz
fi

if [ -n "$DEV_SET" ]; then
    ln -s ${DEV_NPZ} ${MODELDIR}/${ELMO_FILE_PREFIX}-dev.npz
fi

source ${PRJ_DIR}/config/locations.sh

FASTTEXT_NPZ=${FASTTEXT_NPZ_DIR}/fasttext-${LANG_CODE}.npz

PARSER_NAME=udpipe-future
PARSER_DIR=${UDPIPE_FUTURE_DIR}

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

echo ${LANG_CODE} > elmo-lcode.txt
echo ${EMB_CODE} > embcode.txt
ln -s ${FASTTEXT_NPZ} fasttext.npz

if [ -n "$TEST_SET" ]; then
    ln -s ${REAL_TEST_SET} ${FAKE_TBID}-ud-test.conllu
else
    # The parser complains if there is no test set.
    ln -s ${FAKE_TBID}-ud-train.conllu ${FAKE_TBID}-ud-test.conllu
fi

if [ -n "$DEV_SET" ]; then
    ln -s ${REAL_DEV_SET} ${FAKE_TBID}-ud-dev.conllu
fi

MIN_EPOCH_BATCHES=$(expr ${MIN_EPOCH_SENTENCES} / ${BATCH_SIZE})
echo "Batch size is $BATCH_SIZE" >> training.start
echo "Minimum number of batches in each epoch: $MIN_EPOCH_BATCHES" >> training.start
if [ -e ${PARSER_DIR}/.git ] ; then
    echo "Parser commit:" $(git --git-dir=${PARSER_DIR}/.git describe --always) >> training.start
fi
echo "Monitoring sets: $N_MONITORS" >> training.start

echo "Wrapper command line:" >> training.start
echo "$1" >> training.start
echo "$2" >> training.start
echo "$3" >> training.start
echo "$4" >> training.start
echo "$5" >> training.start
echo "$6" >> training.start
echo "$7" >> training.start
echo "$8" >> training.start
echo "$9" >> training.start
echo "${10}" >> training.start
echo "${11}" >> training.start
echo "${12}" >> training.start

touch parser-training.start

python ${PARSER_DIR}/ud_parser.py \
    ${EXTRA_OPTIONS}              \
    --elmo ${ELMO_FILE_PREFIX}    \
    --embeddings fasttext.npz     \
    --seed ${SEED}                \
    --logdir ./                   \
    --batch_size ${BATCH_SIZE}               \
    --min_epoch_batches ${MIN_EPOCH_BATCHES}  \
    --epochs "${LR_SCHEDULE}"     \
    ${FAKE_TBID}                  \
    2> stderr.txt     \
    > stdout.txt

#    --min_epoch_batches 3000      \    ## --> configure MIN_EPOCH_SENTENCES
#    --epochs "4:1e-3,2:1e-4"      \


touch training.end

cd /

if [ -e "$MODELDIR/checkpoint-inference-last.index" ]; then
    rm $MODELDIR/${ELMO_FILE_PREFIX}*.npz
    mv $MODELDIR $FINAL_MODELDIR
else
    SUFFIX=$(head -c 80 /dev/urandom | tr -dc 'A-Za-z0-9' | head -c 12)
    mv $MODELDIR ${FINAL_MODELDIR}-incomplete-${SUFFIX}
fi

if [ -n "$UDPIPE_FUTURE_DELETE_INPUT_NPZ" ]; then
   rm -f "$TRAIN_NPZ" "$TEST_NPZ" "$DEV_NPZ"
fi

