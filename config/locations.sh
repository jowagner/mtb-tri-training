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
SETTING=${USER}@${SIMPLEHOST}
if [ -e /etc/os-release ]; then
    source /etc/os-release
    SETTING=${SETTING}-${ID}-${VERSION_ID}
fi

case "${SETTING}" in
"jwagner@n1")
    export PRJ_DIR=${HOME}/tri-training/mtb-tri-training
    export UD_TREEBANK_DIR=${HOME}/data/ud-treebanks-v2.3
    export CONLL2017_DIR=${HOME}/data/conll2017/clean
    export UDPIPE_FUTURE_DIR=${HOME}/tri-training/UDPipe-Future
    export CONDA_HOME=/ichec/packages/conda/2
    export UDPIPE_FUTURE_ICHEC_CONDA=udpf
    export UDPIPE_FUTURE_LIB_PATH="/ichec/packages/cuda/10.0/lib64":"$HOME/cudnn-for-10.0/lib64"
    export CONLLU_COMBINER_DIR=${HOME}/tri-training/ADAPT-DCU/combination
    export FASTTEXT_NPZ_DIR=${HOME}/data/conll2017/text
    export EFML_TOOL_DIR=${HOME}/tri-training/ELMoForManyLangs
    export EFML_ICHEC_CONDA=efml
    export EFML_LIB_PATH="/ichec/packages/cuda/10.1.243/lib64":"$HOME/cudnn-for-10.1/lib64"
    export EFML_MODEL_DIR=${HOME}/elmo
    export EFML_CACHE_DIR=${HOME}/data/elmo-cache
    export EFML_MAX_CACHE_ENTRIES=100
    ;;
"jwagner@bo-scientific-7.6")
    export PRJ_DIR=${HOME}/tri-training/mtb-tri-training
    export UD_TREEBANK_DIR=${HOME}/data/ud-treebanks-v2.3
    export CONLL2017_DIR=${HOME}/data/conll2017/clean
    export UDPIPE_FUTURE_DIR=${HOME}/tri-training/UDPipe-Future
    export UDPIPE_FUTURE_LIB_PATH=/home/support/nvidia/cuda10/lib64:/home/support/nvidia/cudnn/cuda10_cudnn7_7.5/lib64
    export CONDA_HOME=${HOME}/anaconda3
    export UDPIPE_FUTURE_CONDA=udpf
    export CONLLU_COMBINER_DIR=${HOME}/tri-training/ADAPT-DCU/combination
    export FASTTEXT_NPZ_DIR=${HOME}/data/conll2017/text
    export EFML_TOOL_DIR=${HOME}/tri-training/ELMoForManyLangs
    export EFML_ENV=${HOME}/tri-training/ELMoForManyLangs/venv-efml
    export EFML_MODEL_DIR=${HOME}/elmo
    export EFML_CACHE_DIR=${HOME}/data/elmo-cache
    export EFML_MAX_CACHE_ENTRIES=100
    ;;
"jwagner@ok-opensuse-leap-15.1")
    SCRATCH=/scratch/${USER}
    export PRJ_DIR=${HOME}/tri-training/mtb-tri-training
    export UD_TREEBANK_DIR=${SCRATCH}/ud-parsing/ud-treebanks-v2.3
    export CONLL2017_DIR=${HOME}/data/CoNLL-2017/clean
    export UDPIPE_FUTURE_DIR=${HOME}/tri-training/UDPipe-Future
    export UDPIPE_FUTURE_ENV=${UDPIPE_FUTURE_DIR}/venv-udpf
    export CONLLU_COMBINER_DIR=${HOME}/tbemb/ADAPT-DCU/combination
    export FASTTEXT_NPZ_DIR=${HOME}/data/CoNLL-2017/text
    export EFML_TOOL_DIR=${HOME}/tri-training/ELMoForManyLangs
    export EFML_ENV=${EFML_TOOL_DIR}/venv-efml
    export EFML_MODEL_DIR=${HOME}/elmo
    export EFML_CACHE_DIR=${SCRATCH}/elmo/cache
    export EFML_MAX_CACHE_ENTRIES=20
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
    export CONLL2017_DIR=${HOME}/data/conll2017
    export UDPIPE_FUTURE_DIR=${HOME}/tri-training/UDPipe-Future
    export UDPIPE_FUTURE_LIB_PATH=/home/support/nvidia/cuda10/lib64:/home/support/nvidia/cudnn/cuda10_cudnn7_7.5/lib64
    export CONDA_HOME=${HOME}/anaconda3
    export UDPIPE_FUTURE_CONDA=udpf
    export CONLLU_COMBINER_DIR=${HOME}/tbemb/ADAPT-DCU/combination
    export FASTTEXT_NPZ_DIR=${HOME}/data/UDPipe-Future
    export EFML_TOOL_DIR=${HOME}/tri-training/ELMoForManyLangs
    export EFML_ENV=${EFML_TOOL_DIR}/venv-allennlp-py36
    export EFML_MODEL_DIR=${HOME}/elmo
    ;;
"jwagner@g1-debian-10")
    export PRJ_DIR=${HOME}/tri-training/mtb-tri-training
    export UD_TREEBANK_DIR=${HOME}/data/ud-treebanks-v2.3
    export CONLL2017_DIR=${HOME}/data/conll2017/clean
    export UDPIPE_FUTURE_DIR=${HOME}/tri-training/UDPipe-Future
    export UDPIPE_FUTURE_ENV=${UDPIPE_FUTURE_DIR}/venv-udpf
    export CONLLU_COMBINER_DIR=${HOME}/tri-training/ADAPT-DCU/combination
    export FASTTEXT_NPZ_DIR=${HOME}/data/conll2017/text
    export EFML_TOOL_DIR=${HOME}/tri-training/ELMoForManyLangs
    export EFML_ENV=${EFML_TOOL_DIR}/venv-efml
    export EFML_MODEL_DIR=${HOME}/elmo
    export EFML_CACHE_DIR=${HOME}/data/elmo-cache
    export EFML_MAX_CACHE_ENTRIES=100
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

