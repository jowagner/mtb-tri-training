#!/bin/bash

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

SCRIPT_DIR=${PRJ_DIR}/scripts

OUTPUT_DIR=clean

MAX_JOBS=12

mkdir -p ${OUTPUT_DIR}

L=Uyghur

echo "== $L started $(date) =="
mkdir ${OUTPUT_DIR}/${L}
for I in $L/ug-*.xz ; do
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --info $I  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz) &
    # limit number of tasks running in parallel
    while [ `jobs -r | wc -l | tr -d " "` -ge ${MAX_JOBS} ]; do
        sleep 5
    done
done
wait

L=Irish

echo "== $L started $(date) =="
mkdir ${OUTPUT_DIR}/${L}
for I in $L/ga-*.xz ; do
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --info $I  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz) &
    # limit number of tasks running in parallel
    while [ `jobs -r | wc -l | tr -d " "` -ge ${MAX_JOBS} ]; do
        sleep 5
    done
done
wait

L=Hungarian

echo "== $L started $(date) =="
mkdir ${OUTPUT_DIR}/${L}
for I in $L/hu-*.xz ; do
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --info $I  \
        --fraction 0.12  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz) &
    # limit number of tasks running in parallel
    while [ `jobs -r | wc -l | tr -d " "` -ge ${MAX_JOBS} ]; do
        sleep 5
    done
done
wait

L=Vietnamese

echo "== $L started $(date) =="
mkdir ${OUTPUT_DIR}/${L}
for I in $L/vi-*.xz ; do
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --info $I  \
        --fraction 0.06  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz) &
    # limit number of tasks running in parallel
    while [ `jobs -r | wc -l | tr -d " "` -ge ${MAX_JOBS} ]; do
        sleep 5
    done
done
wait

L=English

echo "== $L started $(date) =="
mkdir ${OUTPUT_DIR}/${L}
for I in $L/en-*.xz ; do
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --info $I  \
        --fraction 0.02  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz) &
    # limit number of tasks running in parallel
    while [ `jobs -r | wc -l | tr -d " "` -ge ${MAX_JOBS} ]; do
        sleep 5
    done
done
wait

echo "== finished $(date) =="
