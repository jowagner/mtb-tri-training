#!/bin/bash

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

test -z $1 && echo "Missing training conllu file"
test -z $1 && exit 1
TRAIN_CONLLU=$1

test -z $2 && echo "Missing language code"
test -z $2 && exit 1
LANG_CODE=$2

test -z $3 && echo "Missing output folder"
test -z $3 && exit 1
OUTPUTDIR=$3

test -z $4 && echo "Missing output filename"
test -z $4 && exit 1
HDF5_NAME=$4

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

source ${PRJ_DIR}/config/locations.sh

if [ -n "$EFML_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=${EFML_LIB_PATH}:${LD_LIBRARY_PATH}
fi

if [ -n "$EFML_CONDA" ]; then
    source ${CONDA_HOME}/etc/profile.d/conda.sh
    conda activate ${EFML_CONDA}
fi

if [ -n "$EFML_ENV" ]; then
    source ${EFML_ENV}/bin/activate
fi

hostname > ${OUTPUTDIR}/elmo.start

LAYER=-1    # (0 for word encoder. LSTM layers: 1 for first, 2 for second, -1 for an average of 3 all layers and -2 for all 3 layers).

INFILE=$(realpath ${TRAIN_CONLLU})
MODEL=${EFML_MODEL_DIR}/${LANG_CODE}_model

test -e ${MODEL} || echo "Missing ${MODEL}"
test -e ${MODEL} || exit 1

# shorten the string as the tool will append ".ly${LAYER}.hdf5 afterwards"
TMP_TOKREPFILE=${OUTPUTDIR}/$(basename ${HDF5_NAME} .hdf5)

# echo "Running ELMoForManyLangs on ${INFILE}"
cd ${EFML_TOOL_DIR}
python -m elmoformanylangs test  \
    --input_format conll           \
    --input ${INFILE}                \
    --model ${MODEL}                  \
    --output_prefix ${TMP_TOKREPFILE}  \
    --output_format hdf5               \
    --output_layer ${LAYER}

mv ${TMP_TOKREPFILE}*ly${LAYER}.hdf5 ${OUTPUTDIR}/${HDF5_NAME}

touch ${OUTPUTDIR}/training.end

