#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8


DATA_DIR=./data-bin/ggw


function glue_func {
    MODEL_DIR=$1
    CKPT=$2
    BATCH_SIZE=$3
    BEAM_SIZE=$4
    MIN_LEN=$5
    MAX_LEN=$6
    NGRAM_SIZE=$7
    LEN_PEN=$8

    CKPT_ID=$(echo ${CKPT} | sed 's/checkpoint//g' | sed 's/\.pt//g' | sed 's/^_//g')

    MODEL=${MODEL_DIR}/${CKPT}
    RESULT_DIR=${MODEL_DIR}/Result/${CKPT_ID}

    mkdir -p ${RESULT_DIR}

    fairseq-generate ${DATA_DIR} --path ${MODEL} \
        --user-dir model --task man \
        --batch-size ${BATCH_SIZE} --beam ${BEAM_SIZE} --min-len ${MIN_LEN} --no-repeat-ngram-size ${NGRAM_SIZE} \
        --lenpen ${LEN_PEN} \
    > ${RESULT_DIR}/res.txt

    cat ${RESULT_DIR}/res.txt | grep "H-" | cut -f3- | sed 's/ ##//g' > ${RESULT_DIR}/hypo.txt
    cat ${RESULT_DIR}/res.txt | grep "T-" | cut -f2- | sed 's/ ##//g' > ${RESULT_DIR}/ref.txt

    files2rouge ${RESULT_DIR}/hypo.txt ${RESULT_DIR}/ref.txt

}


MODEL_DIR=./log/gigaword/man_base_v1
CKPT=checkpoint_best.pt
BATCH_SIZE=128
BEAM_SIZE=5
MIN_LEN=1
MAX_LEN=30
NGRAM_SIZE=3
LEN_PEN=1.0


echo MODEL_DIR: ${MODEL_DIR}. CKPT: ${CKPT}

glue_func ${MODEL_DIR} ${CKPT} ${BATCH_SIZE} ${BEAM_SIZE} ${MIN_LEN} ${MAX_LEN} ${NGRAM_SIZE} ${LEN_PEN}
