#!/bin/bash

# (C) 2019, 2021 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

test -z $1 && echo "Missing training conllu file"
test -z $1 && exit 1
TRAIN_CONLLU=$1

test -z $2 && echo "Missing output folder"
test -z $2 && exit 1
OUTPUTDIR=$2

test -z $3 && echo "Missing output filename"
test -z $3 && exit 1
HDF5_NAME=$3

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

source ${PRJ_DIR}/config/locations.sh

cachelog()
{
    echo `date +%Y-%m-%dT%H:%M:%S` "["$(hostname):$$"]" $* >> ${MBERT_HDF5_CACHE_DIR}/log
}

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

hostname > ${OUTPUTDIR}/mbert.start

LAYER=0    # TODO: double check this is the top layer

INFILE=$(realpath ${TRAIN_CONLLU})

TMP_OUTFILE=${OUTPUTDIR}/${HDF5_NAME}_part

echo "Running mBERT on ${INFILE} to produce ${TMP_OUTFILE}"
cd ${PRJ_DIR}
if ! python scripts/mbert-encode.py \
    --input-format conll            \
    --output-layer ${LAYER}         \
    ${INFILE}                       \
    ${TMP_OUTFILE}
then
    echo "An error occured"
    exit 1
fi

mv ${TMP_OUTFILE} ${OUTPUTDIR}/${HDF5_NAME}

touch ${OUTPUTDIR}/mbert.end

