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

EXTRA_OPTIONS="$4"

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

source ${PRJ_DIR}/config/locations.sh

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

WORKDIR=$(realpath "${OUTPUT}")-workdir

REAL_MODELDIR=$(realpath $MODELDIR)
REAL_WORKDIR=$(realpath $WORKDIR)
REAL_INPUT=$(realpath $INPUT)
REAL_OUTPUT=$(realpath $OUTPUT)

mkdir -p "${WORKDIR}"
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

if [ ${REAL_INPUT: -4} == ".bz2" ]; then
    bzcat ${REAL_INPUT} > input.conllu
    INPUT=input.conllu
else
    INPUT=${REAL_INPUT}
fi

OUTPUT=output.conllu

python ${PARSER_DIR}/ud_parser.py  \
    ${EXTRA_OPTIONS}                \
    --predict                        \
    --predict_input "${INPUT}"    \
    --predict_output "${OUTPUT}"    \
    --embeddings fasttext.npz             \
    --logdir ./                            \
    --checkpoint checkpoint-inference-last  \
    ${FAKE_TBID}                            \
    2> "${REAL_OUTPUT}"-stderr.txt          \
    > stdout.txt

if [ -e "${OUTPUT}" ]; then
    if [ ${REAL_OUTPUT: -4} == ".bz2" ]; then
        bzip2 < ${OUTPUT} > ${REAL_OUTPUT}.part
    else
        mv ${OUTPUT} ${REAL_OUTPUT}.part
    fi
    mv ${REAL_OUTPUT}.part ${REAL_OUTPUT}
fi

cd /
rm -rf ${REAL_WORKDIR}

if [ -e "${REAL_OUTPUT}" ]; then
    rm "${REAL_OUTPUT}"-stderr.txt
fi

