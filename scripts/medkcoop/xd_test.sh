#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/your/datasets"
DATASET=$1
SEED=$2
TRAINER=MeDKCoOp
MODEL=MeDKCoOp
CFG=vit_b16_base2new

SHOTS=16
Device_id=9

DIR=/data/CrossData/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"

else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${MODEL}/few_shot/${DATASET}.yaml \
    --output-dir ${DIR} \
    --model-dir /data/CrossData/BTMRI/${TRAINER}/${CFG}_${SHOTS}shots/${MODEL}/seed${SEED} \
    --mode test \
    --device-id ${Device_id} \
    --load-epoch 100 \
    --eval-only
fi