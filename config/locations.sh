# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner
# Based on locations.sh of https://github.com/tira-io/ADAPT-DCU

# This file must be used with "source bin/activate" from bash or another
# script.


# Reduce hostname to first 2 characters as we don't want to write a
# configuration for each node on the cluster.
SIMPLEHOST=`echo ${HOSTNAME} | cut -c-2 | tr '23456789' '11111111'`

# Need detail on the OS to dinstinguish old and new grove cluster
source /etc/os-release
SETTING=${USER}@${SIMPLEHOST}
if [ -e /etc/os-release ]; then
    source /etc/os-release
    SETTING=${SETTING}-${ID}-${VERSION_ID}
fi

case "${SETTING}" in
"jwagner@n1")
    export UD_TREEBANK_DIR=/ichec/work/dcu01/jwagner/ud-parsing/ud-treebanks-v2.3
    ;;
"jwagner@bo-scientific-7.6")
    export PRJ_DIR=${HOME}/tri-training/mtb-tri-training
    export UD_TREEBANK_DIR=${HOME}/data/ud-treebanks-v2.3
    export UDPIPE_FUTURE_DIR=${HOME}/tri-training/UDPipe-Future
    export UDPIPE_FUTURE_LIB_PATH=/home/support/nvidia/cuda10/lib64:/home/support/nvidia/cudnn/cuda10_cudnn7_7.5/lib64
    export CONDA_HOME=${HOME}/anaconda3
    export UDPIPE_FUTURE_CONDA=udpf
    export CONLLU_COMBINER_DIR=${HOME}/tri-training/ADAPT-DCU/combination
    ;;
"jwagner@ok-opensuse-leap-15.1")
    export PRJ_DIR=${HOME}/tri-training/mtb-tri-training
    export UD_TREEBANK_DIR=/scratch/jwagner/ud-parsing/ud-treebanks-v2.3
    export UDPIPE_FUTURE_DIR=${HOME}/bert/UDPipe-Future
    export UDPIPE_FUTURE_ENV=${UDPIPE_FUTURE_DIR}/venv-tf114-py36
    export CONLLU_COMBINER_DIR=${HOME}/tbemb/ADAPT-DCU/combination
    ;;
"jwagner@g0-debian-9")
    echo "CPU nodes not supported"
    exit 1
    ;;
"jwagner@g0-debian-10")
    echo "CPU nodes not supported"
    exit 1
    ;;
"jwagner@g1-debian-9")
    export PRJ_DIR=${HOME}/tri-training/mtb-tri-training
    export UD_TREEBANK_DIR=${HOME}/tbemb/ud-treebanks-v2.3
    export UDPIPE_FUTURE_DIR=${HOME}/tri-training/UDPipe-Future
    export UDPIPE_FUTURE_LIB_PATH=/home/support/nvidia/cuda10/lib64:/home/support/nvidia/cudnn/cuda10_cudnn7_7.5/lib64
    export CONDA_HOME=${HOME}/anaconda3
    export UDPIPE_FUTURE_CONDA=udpf
    export CONLLU_COMBINER_DIR=${HOME}/tbemb/ADAPT-DCU/combination
    ;;
"jwagner@g1-debian-10")
    export PRJ_DIR=${HOME}/tri-training/mtb-tri-training
    export UD_TREEBANK_DIR=${HOME}/data/ud-treebanks-v2.3
    export UDPIPE_FUTURE_DIR=${HOME}/tri-training/UDPipe-Future
    export UDPIPE_FUTURE_LIB_PATH=/home/support/nvidia/cuda10/lib64:/home/support/nvidia/cudnn/cuda10_cudnn7_7.5/lib64
    export UDPIPE_FUTURE_ENV=${UDPIPE_FUTURE_DIR}/venv-tf114-py36
    export CONLLU_COMBINER_DIR=${HOME}/tri-training/ADAPT-DCU/combination
    ;;
root*)
    # inside udocker
    export PRJ_DIR=/tri-training/mtb-tri-training
    export UD_TREEBANK_DIR=/data/ud-treebanks-v2.3
    ;;
*)
    # default config
    export UD_TREEBANK_DIR=${HOME}/ud-treebanks-v2.3
    ;;
esac

