#!/usr/bin/env bash

ulimit -n 16384
ulimit -n

CODE_PATH=.
cd ${CODE_PATH}

pwd
export PYTHONPATH=${CODE_PATH}:$PYTHONPATH

PROBLEM=cnndm
DATA_DIR=./data-bin/ggw
ARCH=man_base
VERSION=$1
USER_DIR=./model

SAVE_DIR=./log/${PROBLEM}/${ARCH}_v${VERSION}

echo PROBLEM: ${PROBLEM}
echo ARCH: ${ARCH}
echo SAVE_DIR: ${SAVE_DIR}

mkdir -p ${SAVE_DIR}

fairseq-train ${DATA_DIR} --seed 1 \
    --task man --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --arch ${ARCH} \
    --lr 8e-4 \
    --optimizer adam --adam-betas "(0.9,0.98)" --adam-eps 1e-6 --weight-decay 0.01 \
    --clip-norm 0.1 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 8000 \
    --dropout 0.1 --attention-dropout 0.1 \
    --max-tokens 12288 --update-freq 3 \
    --max-source-positions 512 --max-target-positions 512 \
    --skip-invalid-size-inputs-valid-test \
    --num-workers 4 --ddp-backend no_c10d \
    --save-dir ${SAVE_DIR} \
    --max-update 50000 --log-format simple --log-interval 1000 \
    --user-dir ${USER_DIR} \
| tee -a ${SAVE_DIR}/train_log.txt
