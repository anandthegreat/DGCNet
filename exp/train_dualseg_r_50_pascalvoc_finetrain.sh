#!/usr/bin/env bash

# train the net (suppose 8 gpus)
python train_distribute.py \
--data_dir "/home/anand17218/GALD-DGCNet/dataset/PASCALVOC2012/VOCdevkit/VOC2012" \
--arch DualSeg_res50 \
--input_size 832 \
--batch_size_per_gpu 1 \
--learning_rate 0.01 \
--num_steps 50000 \
--num_classes 20 \
--save_dir "./save_dualseg_r50/pascalvoc" \
--ohem 1 --ohem_thres 0.7 --ohem_keep 100000 \
--log_file "dual_seg_r50.log"
