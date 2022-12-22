#!/bin/sh
CUDA_VISIBLE_DEVICES=2  python ./model/main.py --calibrated_stacking 0.4 \
--dataset SUN --cuda --nepoch 300 \
--pretrain_epoch 24 --pretrain_lr 5e-4 --classifier_lr 1e-6 --manualSeed 2347 \
--xe 1 --attri 1e-4 --regular 1e-3  \
--l_xe 1 --l_attri 5e-2 --l_regular 5e-3  \
--consistency 10.0 --consistency-rampup 10 --con 1000 \
--avg_pool --use_group --cpt 2e-7 --gzsl --ins_temp 5 --train_mode distributed  --ways 12  --shots 12  --contrative_loss_weight 1 \
