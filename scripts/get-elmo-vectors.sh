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

cachelog()
{
    echo `date +%Y-%m-%dT%H:%M:%S` "["$(hostname):$$"]" $* >> ${EFML_CACHE_DIR}/log
}

if [ -n "$EFML_CACHE_DIR" ]; then
    CACHE_ENTRY=${LANG_CODE}-$(sha256sum ${TRAIN_CONLLU} | cut -c-64)
    if [ -e "${EFML_CACHE_DIR}/${CACHE_ENTRY}.size" ]; then
        CACHE_ENTRY_SIZE=$(cat ${EFML_CACHE_DIR}/${CACHE_ENTRY}.size)
        cp --reflink=auto ${EFML_CACHE_DIR}/${CACHE_ENTRY}.hdf5 \
            ${OUTPUTDIR}/${HDF5_NAME} 2> /dev/null
	if [ -e ${OUTPUTDIR}/${HDF5_NAME} ]; then
            SIZE=$(wc -c ${OUTPUTDIR}/${HDF5_NAME} | cut -d' ' -f1)
            if [ "$SIZE" == "$CACHE_ENTRY_SIZE" ]; then
                # update last usage information
                touch ${EFML_CACHE_DIR}/${CACHE_ENTRY}.hdf5
                # all done
                cachelog "${CACHE_ENTRY} hit"
                exit 0
            else
                # cannot use file with wrong size
                cachelog "${CACHE_ENTRY} wrong size ${SIZE}, expected ${CACHE_ENTRY_SIZE}, cleaning up"
                rm ${EFML_CACHE_DIR}/${CACHE_ENTRY}.*
            fi
        else
            # cannot use cache entry with missing data
            cachelog "${CACHE_ENTRY} is missing data, cleaning up"
            rm ${EFML_CACHE_DIR}/${CACHE_ENTRY}.*
        fi
    else
        cachelog "${CACHE_ENTRY} miss"
    fi
fi

if [ -n "$EFML_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=${EFML_LIB_PATH}:${LD_LIBRARY_PATH}
fi

if [ -n "$EFML_ICHEC_CONDA" ]; then
    module load conda/2
    source activate ${UDPIPE_FUTURE_ICHEC_CONDA}
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

if [ -n "$EFML_CACHE_DIR" ]; then
    # add output to cache
    if [ -e "${EFML_CACHE_DIR}/${CACHE_ENTRY}.size" ]; then
        # a parallel process was faster
        # --> nothing to do
        cachelog "${CACHE_ENTRY} exists, not updating"
    else
        CACHE_ENTRY_SIZE=$(wc -c ${OUTPUTDIR}/${HDF5_NAME} | cut -d' ' -f1)
        cp --reflink=auto ${OUTPUTDIR}/${HDF5_NAME} \
            ${EFML_CACHE_DIR}/${CACHE_ENTRY}.hdf5
        # signal that entry is ready
        echo ${CACHE_ENTRY_SIZE} > ${EFML_CACHE_DIR}/${CACHE_ENTRY}.size
        cachelog "${CACHE_ENTRY} added"
        # don't let cache grow too much
        NUM_FILES=$(find ${EFML_CACHE_DIR}/ -name "*.size" | wc -l)
        if [ "$NUM_FILES" -gt "$EFML_MAX_CACHE_ENTRIES" ]; then
            PICK_FROM=$(expr ${EFML_MAX_CACHE_ENTRIES} / 2)
            EXPIRED_ENTRY=$(ls -t ${EFML_CACHE_DIR}/ | fgrep .hdf5 |
                tail -n ${PICK_FROM} | shuf | head -n 1)
            EXPIRED_ENTRY=$(basename ${EXPIRED_ENTRY} .hdf5)
            rm ${EFML_CACHE_DIR}/${EXPIRED_ENTRY}.size
            rm ${EFML_CACHE_DIR}/${EXPIRED_ENTRY}.hdf5
            cachelog "${EXPIRED_ENTRY} expired, cleaned up"
        fi
    fi
fi

touch ${OUTPUTDIR}/elmo.end

