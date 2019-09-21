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

SETTING=${USER}@${SIMPLEHOST}

case "${SETTING}" in
"jwagner@li")
    export UD_TREEBANK_DIR=/home/jwagner/Documents/research/parsing/tri-training-2019/ud-treebanks-v2.3
    export UDPIPE_FUTURE_DIR=${HOME}/bert/UDPipe-Future
    ;;
"jwagner@lo")
    export UD_TREEBANK_DIR=/ichec/work/dcu01/jwagner/ud-parsing/ud-treebanks-v2.3
    ;;
"jwagner@n1")
    export UD_TREEBANK_DIR=/ichec/work/dcu01/jwagner/ud-parsing/ud-treebanks-v2.3
    ;;
"jwagner@bo")
    export PRJ_DIR=${HOME}/tri-training/mtb-tri-training
    export UD_TREEBANK_DIR=${HOME}/data/ud-treebanks-v2.3
    export UDPIPE_FUTURE_DIR=${HOME}/tri-training/UDPipe-Future
    export UDPIPE_FUTURE_LIB_PATH=/home/support/nvidia/cuda10/lib64:/home/support/nvidia/cudnn/cuda10_cudnn7_7.5/lib64
    export CONDA_HOME=${HOME}/anaconda3
    export UDPIPE_FUTURE_CONDA=udpf
    export CONLLU_COMBINER_DIR=${HOME}/tri-training/ADAPT-DCU/combination
    ;;
"jwagner@ok")
    export PRJ_DIR=${HOME}/tri-training/mtb-tri-training
    export UD_TREEBANK_DIR=/scratch/jwagner/ud-parsing/ud-treebanks-v2.3
    export UDPIPE_FUTURE_DIR=${HOME}/bert/UDPipe-Future
    export UDPIPE_FUTURE_ENV=${UDPIPE_FUTURE_DIR}/venv-tf114-py36
    export CONLLU_COMBINER_DIR=${HOME}/tbemb/ADAPT-DCU/combination
    ;;
"jwagner@gr")
    export UD_TREEBANK_DIR=${HOME}/tbemb/ud-treebanks-v2.3
    ;;
"jwagner@n0")
    export UD_TREEBANK_DIR=${HOME}/tbemb/ud-treebanks-v2.3
    ;;
"jwagner@g0")
    export UD_TREEBANK_DIR=${HOME}/tbemb/ud-treebanks-v2.3
    ;;
"jwagner@g1")
    export UD_TREEBANK_DIR=${HOME}/tbemb/ud-treebanks-v2.3
    ;;
"jbarry@gr")
    export UD_TREEBANK_DIR=${HOME}/ud-treebanks-v2.3
    ;;
"jbarry@n0")
    export UD_TREEBANK_DIR=${HOME}/ud-treebanks-v2.3
    ;;
root*)
    # inside udocker
    export UD_TREEBANK_DIR=/ud-parsing/ud-treebanks-v2.3
    ;;
*)
    # default config; should work TIRA and for James
    export UD_TREEBANK_DIR=${HOME}/ud-treebanks-v2.3
    ;;
esac

