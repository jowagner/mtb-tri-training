#!/bin/bash

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

SCRIPT_DIR=${PRJ_DIR}/scripts

OUTPUT_DIR=text

mkdir -p ${OUTPUT_DIR}

L=Uyghur

mkdir ${OUTPUT_DIR}/${L}
for I in $L/*.xz ; do
    echo == $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
        >> ${OUTPUT_DIR}/${L}.txt
done

L=Irish

mkdir ${OUTPUT_DIR}/${L}
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

mkdir ${OUTPUT_DIR}/${L}
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

mkdir ${OUTPUT_DIR}/${L}
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

mkdir ${OUTPUT_DIR}/${L}
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

