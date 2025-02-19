# !/usr/bin/env bash
# vs. EXP26
# num layer = 4 + 4
# end ratio = 0.3

src=de
tgt=en

level=subword

ROOT=/apdcephfs/share_916081/khalidcyang
REPO=${ROOT}/project/Latent-GLAT

NAME=${src}-${tgt}
EXP=${ROOT}/project/Latent-GLAT
DATA_BIN=${EXP}/dataset/${level}/bin/${NAME}
LOG_DIR=${EXP}/logs/${NAME}/${level}
SAVE_DIR=${EXP}/checkpoints/${NAME}/${level}


MODEL=latent-GLAT-raw

mkdir -p ${SAVE_DIR}/${MODEL}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 ${REPO}/train.py \
    ${DATA_BIN} \
    --user-dir ${REPO}/latent_glat \
    --tensorboard-logdir ${LOG_DIR}/${MODEL} \
    --save-dir ${SAVE_DIR}/${MODEL} \
    --task nat \
    --ddp-backend=no_c10d \
    --criterion generic_loss \
    --arch vqnat_wmt14 \
    --share-all-embeddings \
    --mapping-func interpolate \
    --mapping-use output \
    --share-rel-embeddings \
    --block-cls highway \
    --self-attn-cls shaw \
    --enc-self-attn-cls shaw --enc-block-cls highway \
    --max-rel-positions 4 \
    --noise full_mask \
    --apply-bert-init \
    --optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 10.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --min-lr 1e-09 --weight-decay 0.0 --dropout 0.1 \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --label-smoothing 0.0 \
    --log-format 'simple' --log-interval 100 \
    --max-tokens 4000 \
    --update-freq 1 \
    --save-interval-updates 500 \
    --keep-best-checkpoints 5 \
    --no-epoch-checkpoints \
    --keep-interval-updates 5 \
    --max-update 300000 \
    --num-workers 0 \
    --eval-bleu \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --no-accuracy \
    --left-pad-source False \
    --glat-training \
    --start-ratio 0.5 \
    --end-ratio 0.3 \
    --anneal-start 4000 \
    --anneal-steps 150000 \
    --print-ratio-every 300000 \
    --latent-factor 1.0 \
    --vq-ema --gated-func residual \
    --vq-glat \
    --vq-schedule-ratio 0.5 \
    --vq-mix-diff \
    --latent-layers 4 \
    --decoder-layers 4 \
    --vq-end-ratio 0.3 \
    --num-codes 64 >> ${SAVE_DIR}/${MODEL}/train.log 2>&1

    # --max-sentences-valid 1 \