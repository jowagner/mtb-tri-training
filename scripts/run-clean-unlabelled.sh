#!/bin/bash

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

SCRIPT_DIR=${PRJ_DIR}/scripts

OUTPUT_DIR=clean

mkdir -p ${OUTPUT_DIR}

L=Uyghur

mkdir ${OUTPUT_DIR}/${L}
for I in $L/*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done

L=Irish

mkdir ${OUTPUT_DIR}/${L}
for I in $L/??-common*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --fraction 0.33074  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done
for I in $L/??-wiki*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done

L=Hungarian

mkdir ${OUTPUT_DIR}/${L}
for I in $L/??-common*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --fraction 0.00495  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done
for I in $L/??-wiki*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --fraction 0.03576  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done

L=Vietnamese

mkdir ${OUTPUT_DIR}/${L}
for I in $L/??-common*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --fraction 0.00209  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done
for I in $L/??-wiki*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --fraction 0.05440  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done

L=English

mkdir ${OUTPUT_DIR}/${L}
for I in $L/??-common*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --fraction 0.00115  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done
for I in $L/??-wiki*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --fraction 0.00205  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done

