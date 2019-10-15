#!/bin/bash

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

test -z $1 && echo "Missing model folder"
test -z $1 && exit 1
MODELDIR=$1

test -z $2 && echo "Missing conllu input file"
test -z $2 && exit 1
INPUT=$2

test -z $3 && echo "Missing conllu output file"
test -z $3 && exit 1
OUTPUT=$3

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

WORKDIR=$(realpath "${OUTPUT}")-workdir
mkdir -p "${WORKDIR}"

LANG_CODE=$(cat ${MODELDIR}/elmo-lcode.txt)

ELMO_FILE_PREFIX=elmo

${PRJ_DIR}/scripts/get-elmo-vectors.sh  \
    ${INPUT}                            \
    ${LANG_CODE}                        \
    ${WORKDIR}                         \
    ${ELMO_FILE_PREFIX}-test.hdf5       \
    2> ${WORKDIR}/elmo-stderr-test.txt  \
    >  ${WORKDIR}/elmo-stdout-test.txt

${PRJ_DIR}/scripts/elmo-hdf5-to-npz.py       \
    --elmoformanylang ${INPUT}                \
    ${WORKDIR}/${ELMO_FILE_PREFIX}-test.hdf5  \
    ${WORKDIR}/${ELMO_FILE_PREFIX}-test.npz

rm ${WORKDIR}/${ELMO_FILE_PREFIX}-test.hdf5

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

REAL_MODELDIR=$(realpath $MODELDIR)
REAL_INPUT=$(realpath $INPUT)
REAL_OUTPUT=$(realpath $OUTPUT)

cd ${WORKDIR}

for ENTRY in   \
    checkpoint  \
    checkpoint-inference-last.data-00000-of-00001  \
    checkpoint-inference-last.index                \
    fasttext.npz                                   \
    xx_xxx-ud-train.conllu                         \
; do
    ln -s ${REAL_MODELDIR}/${ENTRY}
done

python ${PARSER_DIR}/ud_parser.py  \
    --predict                        \
    --predict_input "${REAL_INPUT}"    \
    --predict_output "${REAL_OUTPUT}"   \
    --elmo ${ELMO_FILE_PREFIX}-test.npz  \
    --embeddings fasttext.npz             \
    --logdir ./                            \
    --checkpoint checkpoint-inference-last  \
    ${FAKE_TBID}                            \
    2> stderr.txt                           \
    > stdout.txt

cd /
rm -rf ${WORKDIR}

