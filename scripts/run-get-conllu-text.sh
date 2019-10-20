#!/bin/bash

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/mtb-tri-training

SCRIPT_DIR=${PRJ_DIR}/scripts

OUTPUT_DIR=text

mkdir -p ${OUTPUT_DIR}

NUM_PASSES=100

L=Uyghur

rm ${OUTPUT_DIR}/${L}.txt
for PASS in {1..${NUM_PASSES}} ; do
for I in $L/*.xz ; do
    echo == Pass ${PASS} of ${NUM_PASSES} for $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
	--pass ${PASS} --passes ${NUM_PASSES}  \
        >> ${OUTPUT_DIR}/${L}.txt
done
done

L=Irish

rm ${OUTPUT_DIR}/${L}.txt
for PASS in {1..${NUM_PASSES}} ; do
for I in $L/??-common*.xz ; do
    echo == Pass ${PASS} of ${NUM_PASSES} for $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
	--pass ${PASS} --passes ${NUM_PASSES}  \
        >> ${OUTPUT_DIR}/${L}.txt
done
for I in $L/??-wiki*.xz ; do
    echo == Pass ${PASS} of ${NUM_PASSES} for $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
	--pass ${PASS} --passes ${NUM_PASSES}  \
        >> ${OUTPUT_DIR}/${L}.txt
done
done

L=Hungarian

rm ${OUTPUT_DIR}/${L}.txt
for PASS in {1..${NUM_PASSES}} ; do
for I in $L/??-common*.xz ; do
    echo == Pass ${PASS} of ${NUM_PASSES} for $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
	--pass ${PASS} --passes ${NUM_PASSES}  \
        >> ${OUTPUT_DIR}/${L}.txt
done
for I in $L/??-wiki*.xz ; do
    echo == Pass ${PASS} of ${NUM_PASSES} for $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
	--pass ${PASS} --passes ${NUM_PASSES}  \
        >> ${OUTPUT_DIR}/${L}.txt
done
done

L=Vietnamese

rm ${OUTPUT_DIR}/${L}.txt
for PASS in {1..${NUM_PASSES}} ; do
for I in $L/??-common*.xz ; do
    echo == Pass ${PASS} of ${NUM_PASSES} for $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
	--pass ${PASS} --passes ${NUM_PASSES}  \
        >> ${OUTPUT_DIR}/${L}.txt
done
for I in $L/??-wiki*.xz ; do
    echo == Pass ${PASS} of ${NUM_PASSES} for $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
	--pass ${PASS} --passes ${NUM_PASSES}  \
        >> ${OUTPUT_DIR}/${L}.txt
done
done

L=English

rm ${OUTPUT_DIR}/${L}.txt
for PASS in {1..${NUM_PASSES}} ; do
for I in $L/??-common*.xz ; do
    echo == Pass ${PASS} of ${NUM_PASSES} for $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
	--pass ${PASS} --passes ${NUM_PASSES}  \
        >> ${OUTPUT_DIR}/${L}.txt
done
for I in $L/??-wiki*.xz ; do
    echo == Pass ${PASS} of ${NUM_PASSES} for $I ==
    unxz < $I | \
        ${SCRIPT_DIR}/get-conllu-text.py \
	--pass ${PASS} --passes ${NUM_PASSES}  \
        >> ${OUTPUT_DIR}/${L}.txt
done
done

