#!/bin/bash

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

source ${PRJ_DIR}/config/locations.sh

if [ -n "$EFML_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=${EFML_LIB_PATH}:${LD_LIBRARY_PATH}
fi

if [ -n "$EFML_ICHEC_CONDA" ]; then
    module load conda/2
    source activate ${EFML_ICHEC_CONDA}
fi

if [ -n "$EFML_CONDA" ]; then
    source ${CONDA_HOME}/etc/profile.d/conda.sh
    conda activate ${EFML_CONDA}
fi

if [ -n "$EFML_ENV" ]; then
    source ${EFML_ENV}/bin/activate
fi

python ${PRJ_DIR}/scripts/elmo-hdf5-to-npz.py "$@"

