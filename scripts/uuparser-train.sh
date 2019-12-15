#!/bin/bash

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

test -z $1 && echo "Missing list of TBIDs"
test -z $1 && exit 1
TBIDS=$1

test -z $2 && echo "Missing seed"
test -z $2 && exit 1
SEED=$2

test -z $3 && echo "Missing model output folder"
test -z $3 && exit 1
MODELDIR=$(realpath $3)

if [ -e "$MODELDIR/$TBIDS" ]; then
    echo "Refusing to overwrite model for $TBIDS"
    exit 1
fi

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

source ${PRJ_DIR}/config/locations.sh

MEM=6144 # initial amount of dynet memory; will be increased automatically by dynet if needed
EPOCHS=6

PARSER_NAME=src/parser.py
PARSER_DIR=${UUPARSER_DIR}

if [ -n "$UUPARSER_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=${UUPARSER_LIB_PATH}:${LD_LIBRARY_PATH}
fi

if [ -n "$UUPARSER_ICHEC_CONDA" ]; then
    module load conda/2
    source activate ${UUPARSER_ICHEC_CONDA}
fi

if [ -n "$UUPARSER_CONDA" ]; then
    source ${CONDA_HOME}/etc/profile.d/conda.sh
    conda activate ${UUPARSER_CONDA}
fi

if [ -n "$UUPARSER_ENV" ]; then
    source ${UUPARSER_ENV}/bin/activate
    echo "Activated $UUPARSER_ENV"
fi

mkdir -p ${MODELDIR}/stats

echo $(hostname) $(date) > ${MODELDIR}/stats/training.start

#    --deadline ${DEADLINE} \
#    ${DYNET_OPTIONS} \
#    --top-k-epochs 3  \
#    --fingerprint  \

cd ${PARSER_DIR}

python2 scripts/create_json.py ${MODELDIR}/FAKE_UD_DIR/ ${MODELDIR}/ud_iso.json

python ${PARSER_NAME}  \
    --json-isos ${MODELDIR}/ud_iso.json  \
    --max-sentences 250  \
    --dynet-seed ${SEED}  \
    --dynet-mem ${MEM}  \
    --outdir ${MODELDIR}/${TBIDS}  \
    --modeldir ${MODELDIR}/${TBIDS} \
    --datadir ${MODELDIR}/FAKE_UD_DIR  \
    --include ${TBIDS} \
    --epochs ${EPOCHS}  \
    --k 3      \
    --userl    \
    --multiling \
    2> ${MODELDIR}/${FILE}/stderr.txt \
    >  ${MODELDIR}/${FILE}/stdout.txt

touch ${MODELDIR}/stats/training.end

