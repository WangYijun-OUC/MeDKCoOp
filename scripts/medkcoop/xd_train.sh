#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/your/datasets"
DATASET=$1
TRAINER=MeDKCoOp
MODEL=MeDKCoOp
CFG=vit_b16_base2new
SHOTS=16
Device_id=9

for SEED in 1 2 3
do
DIR=/data/CrossData/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/${MODEL}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."

else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${Device_id} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${MODEL}/few_shot/${DATASET}.yaml \
    --mode train \
    --device-id ${Device_id} \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
done