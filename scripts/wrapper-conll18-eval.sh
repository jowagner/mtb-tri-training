#!/bin/bash

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

source ${PRJ_DIR}/config/locations.sh

if [ -n "$PYTHON3_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=${PYTHON3_LIB_PATH}:${LD_LIBRARY_PATH}
fi

cmd=python3

if [ -n "$PYTHON3_ICHEC_CONDA" ]; then
    module load conda/2
    source activate ${PYTHON3_ICHEC_CONDA}
    cmd=python
fi

if [ -n "$PYTHON3_CONDA" ]; then
    source ${CONDA_HOME}/etc/profile.d/conda.sh
    conda activate ${PYTHON3_CONDA}
    cmd=python
fi

if [ -n "$PYTHON3_ENV" ]; then
    source ${PYTHON3_ENV}/bin/activate
    cmd=python
fi

$cmd ${PRJ_DIR}/scripts/conll18_ud_eval.py "$@"

