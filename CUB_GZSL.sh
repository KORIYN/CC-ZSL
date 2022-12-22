#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python ./model/main.py \
--dataset CUB \
--calibrated_stacking 0.7 --consistency 10.0 --consistency-rampup 10 \
--cuda --nepoch 300  --train_id 0 --manualSeed 3131 \
--pretrain_epoch 21  --pretrain_lr 1e-4 --classifier_lr 1e-6 \
--xe 1 --attri 1e-2 --regular 5e-6 \
--l_xe 1 --l_attri 1e-1  --l_regular 4e-2 \
--cpt 1e-9 --con 100  --ins_temp 1 --train_mode distributed --ways 8  --shots 12  --use_group --gzsl  --contrative_loss_weight 1 \
