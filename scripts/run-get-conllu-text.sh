#!/bin/bash

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

SCRIPT_DIR=${PRJ_DIR}/scripts

OUTPUT_DIR=text

mkdir -p ${OUTPUT_DIR}

L=Uyghur

for I in $L/*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
        >> ${OUTPUT_DIR}/${L}.txt
done

L=Irish

for I in $L/??-common*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
        >> ${OUTPUT_DIR}/${L}.txt
done
for I in $L/??-wiki*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
        >> ${OUTPUT_DIR}/${L}.txt
done

L=Hungarian

for I in $L/??-common*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
        >> ${OUTPUT_DIR}/${L}.txt
done
for I in $L/??-wiki*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
        >> ${OUTPUT_DIR}/${L}.txt
done

L=Vietnamese

for I in $L/??-common*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
        >> ${OUTPUT_DIR}/${L}.txt
done
for I in $L/??-wiki*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
        >> ${OUTPUT_DIR}/${L}.txt
done

L=English

for I in $L/??-common*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
        >> ${OUTPUT_DIR}/${L}.txt
done
for I in $L/??-wiki*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
        >> ${OUTPUT_DIR}/${L}.txt
done

