#!/bin/bash

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

test -z $1 && echo "Missing training conllu file"
test -z $1 && exit 1
TRAIN_CONLLU=$1

test -z $2 && echo "Missing language code"
test -z $2 && exit 1
LANG_CODE=$2

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

# optional args:
DEV_SET=$7
TEST_SET=$8

if [ -n "$TEST_SET" ]; then
    REAL_TEST_SET=$(realpath ${TEST_SET})
fi
if [ -n "$DEV_SET" ]; then
    REAL_DEV_SET=$(realpath ${DEV_SET})
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

${PRJ_DIR}/scripts/get-elmo-vectors.sh  \
    ${TRAIN_CONLLU}                     \
    ${LANG_CODE}                        \
    ${MODELDIR}                         \
    ${ELMO_FILE_PREFIX}-train.hdf5       \
    2> ${MODELDIR}/elmo-stderr-train.txt  \
    >  ${MODELDIR}/elmo-stdout-train.txt

#rm -f ${MODELDIR}/${ELMO_FILE_PREFIX}-train.npz

${PRJ_DIR}/scripts/wrapper-elmo-hdf5-to-npz.sh  \
    --elmoformanylang ${TRAIN_CONLLU}          \
    ${MODELDIR}/${ELMO_FILE_PREFIX}-train.hdf5  \
    ${MODELDIR}/${ELMO_FILE_PREFIX}-train.npz

rm ${MODELDIR}/${ELMO_FILE_PREFIX}-train.hdf5

if [ -n "$TEST_SET" ]; then
    ${PRJ_DIR}/scripts/get-elmo-vectors.sh  \
        ${TEST_SET}                         \
        ${LANG_CODE}                        \
        ${MODELDIR}                         \
        ${ELMO_FILE_PREFIX}-test.hdf5       \
        2> ${MODELDIR}/elmo-stderr-test.txt  \
        >  ${MODELDIR}/elmo-stdout-test.txt
    #rm -f ${MODELDIR}/${ELMO_FILE_PREFIX}-test.npz
    ${PRJ_DIR}/scripts/wrapper-elmo-hdf5-to-npz.sh  \
        --elmoformanylang ${TEST_SET}          \
        ${MODELDIR}/${ELMO_FILE_PREFIX}-test.hdf5  \
        ${MODELDIR}/${ELMO_FILE_PREFIX}-test.npz
    rm ${MODELDIR}/${ELMO_FILE_PREFIX}-test.hdf5
fi

if [ -n "$DEV_SET" ]; then
    ${PRJ_DIR}/scripts/get-elmo-vectors.sh  \
        ${DEV_SET}                          \
        ${LANG_CODE}                        \
        ${MODELDIR}                         \
        ${ELMO_FILE_PREFIX}-dev.hdf5        \
        2> ${MODELDIR}/elmo-stderr-dev.txt  \
        >  ${MODELDIR}/elmo-stdout-dev.txt
    #rm -f ${MODELDIR}/${ELMO_FILE_PREFIX}-dev.npz
    ${PRJ_DIR}/scripts/wrapper-elmo-hdf5-to-npz.sh  \
        --elmoformanylang ${DEV_SET}          \
        ${MODELDIR}/${ELMO_FILE_PREFIX}-dev.hdf5  \
        ${MODELDIR}/${ELMO_FILE_PREFIX}-dev.npz
    rm ${MODELDIR}/${ELMO_FILE_PREFIX}-dev.hdf5
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

cp ${TRAIN_CONLLU} $MODELDIR/${FAKE_TBID}-ud-train.conllu
cd $MODELDIR

echo ${LANG_CODE} > elmo-lcode.txt
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

touch parser-training.start

MIN_EPOCH_BATCHES=$(expr ${MIN_EPOCH_SENTENCES} / ${BATCH_SIZE})
echo "Batch size is $BATCH_SIZE" >> training.start
echo "Minimum number of batches in each epoch: $MIN_EPOCH_BATCHES" >> training.start

python ${PARSER_DIR}/ud_parser.py \
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

rm ${ELMO_FILE_PREFIX}*.npz

touch training.end

cd /
mv $MODELDIR $FINAL_MODELDIR

