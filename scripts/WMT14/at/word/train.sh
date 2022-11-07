#!/usr/bin/env bash

src=de
tgt=en

level=word

ROOT=/apdcephfs/share_916081/khalidcyang
REPO=${ROOT}/project/Latent-GLAT

NAME=${src}-${tgt}
EXP=${ROOT}/project/Latent-GLAT
DATA_BIN=${EXP}/dataset/${level}/bin/${NAME}
LOG_DIR=${EXP}/logs/${NAME}/${level}
SAVE_DIR=${EXP}/checkpoints/${NAME}/${level}


MODEL=vanilla_transformer

mkdir -p ${SAVE_DIR}/${MODEL} 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 ${REPO}/train.py \
    ${DATA_BIN} \
    --source-lang de --target-lang en \
    --arch transformer_wmt_en_de \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 4096 --update-freq 1 \
    --max-update 400000 \
    --save-dir ${SAVE_DIR}/${MODEL} \
    --tensorboard-logdir ${LOG_DIR}/${MODEL} \
    --lr 7e-4 \
    --upsample-primary 1 \
    --share-decoder-input-output-embed \
    --eval-bleu \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --left-pad-source False \
    --save-interval-updates 500 \
    --keep-best-checkpoints 5 \
    --no-epoch-checkpoints \
    --keep-interval-updates 5 \
    --log-format 'simple' --log-interval 100 >> ${SAVE_DIR}/${MODEL}/train.log 2>&1

    # --max-sentences-valid 1 \