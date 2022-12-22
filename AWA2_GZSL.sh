#!/bin/sh
CUDA_VISIBLE_DEVICES=1 python ./model/main.py --dataset AWA2 \
--cuda --nepoch 300 \
--pretrain_epoch 4 --pretrain_lr 1e-4 --classifier_lr 1e-7 --manualSeed 8275 \
--xe 1 --attri 1e-4 --regular 0.0005 \
--l_xe 0 --l_attri 1e-2 --l_regular 0.5e-6 --cpt 2e-9 \
--avg_pool --batch_size 64 \
--consistency 10.0 --consistency-rampup 10 \
--calibrated_stacking 0.7 --all --gzsl --con 1 --ins_temp 1 --train_mode distributed  --ways 8  --shots 12  --contrative_loss_weight 1 \
