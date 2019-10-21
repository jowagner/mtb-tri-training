#!/bin/bash

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

SCRIPT_DIR=${PRJ_DIR}/scripts

OUTPUT_DIR=text3

mkdir -p ${OUTPUT_DIR}

MAX_JOBS=12

for L in \
    Uyghur \
    Irish \
    Hungarian \
    English \
    Vietnamese \
; do
    echo "== $L started $(date) =="
    for I in $L/*.conllu.xz ; do
        TMP=/tmp/$$-$L-$(basename $I .conllu.xz).tsv
        unxz < $I | \
            ${SCRIPT_DIR}/get-conllu-text.py \
            --info $I  \
	    --random-prefix | \
            LC_ALL=C sort -S 1G > ${TMP} &

        # limit number of tasks
        while [ `jobs -r | wc -l | tr -d " "` -ge ${MAX_JOBS} ]; do
            sleep 5
        done
    done
    echo "Waiting for $L jobs to finish..."
    wait
    echo "Merging data for ${L}..."
    LC_ALL=C sort --merge --batch-size=5 /tmp/$$-$L-*.tsv | \
        cut -f2- > ${OUTPUT_DIR}/$L.txt
    echo "Cleaning up..."
    rm -f /tmp/$$-$L-*.tsv
done
echo "== finished $(date) =="

