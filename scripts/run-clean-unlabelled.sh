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
for I in ??-common*.xz ; do
    echo == $L/$I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --fraction 0.25  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done
for I in ??-wiki*.xz ; do
    echo == $L/$I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done

L=Hungarian

mkdir ${OUTPUT_DIR}/${L}
for I in ??-common*.xz ; do
    echo == $L/$I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --fraction 0.02  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done
for I in ??-wiki*.xz ; do
    echo == $L/$I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --fraction 0.10  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done

L=Vietnamese

mkdir ${OUTPUT_DIR}/${L}
for I in ??-common*.xz ; do
    echo == $L/$I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --fraction 0.005  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done
for I in ??-wiki*.xz ; do
    echo == $L/$I ==
    unxz < $I | \
        ${SCRIPT_DIR}/clean-unlabelled-conllu.py \
        --fraction 0.125  \
        > ${OUTPUT_DIR}/${L}/$(basename $I .xz)
done

