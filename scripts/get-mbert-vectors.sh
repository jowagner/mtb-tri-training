#!/bin/bash

# (C) 2019, 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

test -z $1 && echo "Missing training conllu file"
test -z $1 && exit 1
TRAIN_CONLLU=$1

test -z $2 && echo "Missing layer specification"
test -z $2 && exit 1
LAYER=$2

test -z $3 && echo "Missing expand_to specification"
test -z $3 && exit 1
EXPAND_TO=$3

test -z $4 && echo "Missing pooling specification"
test -z $4 && exit 1
POOLING=$4

test -z $5 && echo "Missing output folder"
test -z $5 && exit 1
OUTPUTDIR=$5

test -z $6 && echo "Missing output filename"
test -z $6 && exit 1
HDF5_NAME=$6

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

source ${PRJ_DIR}/config/locations.sh

if [ -n "$MBERT_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=${MBERT_LIB_PATH}:${LD_LIBRARY_PATH}
fi

if [ -n "$MBERT_ICHEC_CONDA" ]; then
    module load conda/2
    source activate ${MBERT_ICHEC_CONDA}
fi

if [ -n "$MBERT_CONDA" ]; then
    source ${CONDA_HOME}/etc/profile.d/conda.sh
    conda activate ${MBERT_CONDA}
fi

if [ -n "$MBERT_ENV" ]; then
    source ${MBERT_ENV}/bin/activate
fi

# ensure output folder exists
mkdir -p ${OUTPUTDIR}

hostname > ${OUTPUTDIR}/mbert.start

if [ -e "$TRAIN_CONLLU" ]; then
    echo "Input file is ready" > /dev/null
else
    echo "Waiting for input file"
    sleep 60
fi

INFILE=$(realpath ${TRAIN_CONLLU})

TMP_OUTFILE=${OUTPUTDIR}/${HDF5_NAME}_part

#if [ -z $CUDA_VISIBLE_DEVICES ] ; then
#    echo "Running cpu-only"
#else
#    echo "Monitoring GPU usage in parallel"
#    #nvidia-smi
#    #nvidia-smi dmon &  # cannot poll faster than 1 second
#    nvidia-smi --loop-ms=3000 &
#    NVIDIA_SMI_PID=$!
#fi

echo "Running mBERT on ${INFILE} to produce ${TMP_OUTFILE}"
cd ${PRJ_DIR}
if ! python scripts/mbert-encode.py \
    ${MBERT_OPTIONS}                \
    --input-format conll            \
    --output-layer ${LAYER}         \
    --expand-to ${EXPAND_TO}        \
    --pooling ${POOLING}            \
    ${INFILE}                       \
    ${TMP_OUTFILE} 2> ${TMP_OUTFILE}.err
then
    echo "An error occured"
    cat ${TMP_OUTFILE}.err
    echo "Repeating the run with --debug to produce detailed log"
    python scripts/mbert-encode.py \
        --debug                         \
        ${MBERT_OPTIONS}                \
        --input-format conll            \
        --output-layer ${LAYER}         \
        --expand-to ${EXPAND_TO}        \
        --pooling ${POOLING}            \
        ${INFILE}                       \
        ${TMP_OUTFILE} | tee ${TMP_OUTFILE}-log.txt
    exit 1
else
    # no error -> clean up
    rm ${TMP_OUTFILE}.err
fi

mv ${TMP_OUTFILE} ${OUTPUTDIR}/${HDF5_NAME}

touch ${OUTPUTDIR}/mbert.end

#if [ -z $CUDA_VISIBLE_DEVICES ] ; then
#    echo
#else
#    kill $NVIDIA_SMI_PID
#    sleep 1
#    echo
#fi
