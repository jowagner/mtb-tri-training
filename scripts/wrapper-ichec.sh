#!/bin/bash
  
WORKDIR=$1
SCRIPTDIR=$2
CUDA_VISIBLE_DEVICES=$3
export CUDA_VISIBLE_DEVICES

shift
shift
shift

mkdir -p ${WORKDIR}

cd ${WORKDIR}
touch run.start
echo $(hostname) $(date) >> run.start
if [ -e run.end ]; then
    mv run.end run.prev-end
fi

cd ${SCRIPTDIR}
./tri-train.py  "$@"         \
    2> ${WORKDIR}/stderr.txt  \
    >  ${WORKDIR}/stdout.txt

touch ${WORKDIR}/run.end

