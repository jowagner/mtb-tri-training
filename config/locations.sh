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
    UD_TREEBANK_DIR=/home/jwagner/Documents/research/parsing/tri-training-2019/ud-treebanks-v2.3
    UDPIPE_FUTURE_DIR=${HOME}/bert/UDPipe-Future
    ;;
"jwagner@lo")
    UD_TREEBANK_DIR=/ichec/work/dcu01/jwagner/ud-parsing/ud-treebanks-v2.3
    ;;
"jwagner@n1")
    UD_TREEBANK_DIR=/ichec/work/dcu01/jwagner/ud-parsing/ud-treebanks-v2.3
    ;;
"jwagner@ok")
    UD_TREEBANK_DIR=/scratch/jwagner/ud-parsing/ud-treebanks-v2.3
    UDPIPE_FUTURE_DIR=${HOME}/bert/UDPipe-Future
    UDPIPE_FUTURE_ENV=${UDPIPE_FUTURE_DIR}/venv-tf114-py36
    ;;
"jwagner@gr")
    UD_TREEBANK_DIR=${HOME}/tbemb/ud-treebanks-v2.3
    ;;
"jwagner@n0")
    UD_TREEBANK_DIR=${HOME}/tbemb/ud-treebanks-v2.3
    ;;
"jwagner@g0")
    UD_TREEBANK_DIR=${HOME}/tbemb/ud-treebanks-v2.3
    ;;
"jwagner@g1")
    UD_TREEBANK_DIR=${HOME}/tbemb/ud-treebanks-v2.3
    ;;
"jbarry@gr")
    UD_TREEBANK_DIR=${HOME}/ud-treebanks-v2.3
    ;;
"jbarry@n0")
    UD_TREEBANK_DIR=${HOME}/ud-treebanks-v2.3
    ;;
root*)
    # inside udocker
    UD_TREEBANK_DIR=/ud-parsing/ud-treebanks-v2.3
    ;;
*)
    # default config; should work TIRA and for James
    UD_TREEBANK_DIR=${HOME}/ud-treebanks-v2.3
    ;;
esac

