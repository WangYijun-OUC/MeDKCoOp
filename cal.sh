#!/bin/bash

#cd ../..

# custom config

# MODEL=$1

for DATASET in CHMNIST covid ctkidney kvasir lungcolon octmnist retina BTMRI1
do
python parse_test_res.py /data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/train_base/${DATASET}/shots_16/GRAPH_DCPL_b08/vit_b16_0002_b50_randflip --test-log
python parse_test_res.py /data/wangyijun/ckpt/Biomed_Graph_dcpl/ablation/test_new/${DATASET}/shots_16/GRAPH_DCPL_b08/vit_b16_0002_b50_randflip --test-log
done