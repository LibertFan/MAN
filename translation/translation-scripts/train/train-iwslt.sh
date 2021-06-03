#!/usr/bin/env bash

CODE_PATH=.
cd ${CODE_PATH}
export PYTHONPATH=$CODE_PATH:$PYTHONPATH


PROBLEM=iwslt_de_en
ARCH=man_iwslt_de_en
VERSION=$1
DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined/
OUTPUT_PATH=log/${PROBLEM}/${ARCH}_v${VERSION}

mkdir -p ${OUTPUT_PATH}

# train on 2 24G P40

python train.py ${DATA_PATH} \
  --seed 1 \
  --arch man_iwslt_de_en --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --dropout 0.3 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 16000 \
  --lr 7e-3 --min-lr 1e-09 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 1e-4 \
  --max-tokens 8192 --save-dir ${OUTPUT_PATH} \
  --update-freq 1 --no-progress-bar --log-interval 50 \
  --ddp-backend no_c10d \
  --save-interval-updates 10000 --keep-interval-updates 5 --max-epoch 250 \
| tee -a ${OUTPUT_PATH}/train_log.txt
