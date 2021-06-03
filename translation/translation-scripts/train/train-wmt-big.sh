#!/usr/bin/env bash

CODE_PATH=.
cd ${CODE_PATH}
export PYTHONPATH=${CODE_PATH}:${PYTHONPATH}

PROBLEM=wmt_en_de
ARCH=man_wmt_en_de_big
VERSION=$1
DATA_PATH=data-bin/wmt14_en_de_joined_dict/
OUTPUT_PATH=log/${PROBLEM}/${ARCH}_${VERSION}

mkdir -p ${OUTPUT_PATH}

# Assume training on 8 V100 GPUs. Change the --max-tokens and --update-freq to match your hardware settings.

python train.py ${DATA_PATH} \
  --seed 1 \
  --arch ${ARCH} --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0  \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
  --lr 0.015 --min-lr 1e-09 --max-update 50000 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 4096 --save-dir ${OUTPUT_PATH} \
  --update-freq 16 --no-progress-bar --log-interval 1000 \
  --ddp-backend no_c10d \
  --save-interval-updates 10000 --keep-interval-updates 20 \
| tee -a ${OUTPUT_PATH}/train_log.txt
