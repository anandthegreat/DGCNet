#!/usr/bin/env bash

# train the net (suppose 8 gpus)
# --restore_from "/home/anand17218/GALD-DGCNet/pretrained/resnet50-pascalvoc.pth" \
python train_distribute.py --data_set pascalvoc \
--data_dir "/home/anand17218/GALD-DGCNet/dataset/PASCALVOC2012/VOCdevkit/VOC2012" \
--arch DualSeg_res50 \
--input_size 321 \
--batch_size_per_gpu 8 \
--learning_rate 0.01 \
--num_steps 250000 \
--num_classes 21 \
--save_dir "./save_dualseg_r50/pascalvoc" \
--ohem 1 --ohem_thres 0.7 --ohem_keep 100000 \
--log_file "dual_seg_r50_pascalvoc.log"

