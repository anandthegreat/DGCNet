#!/usr/bin/env bash

# train the net (suppose 8 gpus)
python -m torch.distributed.launch --nproc_per_node=8 train_distribute.py --data_set cityscapes \
--data_dir "/home/anand17218/GALD-DGCNet/dataset" \
--data_list "./data/cityscapes/train.txt" \
--arch DualSeg_res101 \
--restore_from "/home/anand17218/GALD-DGCNet/pretrained/resnet101-deep.pth" \
--input_size 832 \
--batch_size_per_gpu 1 \
--rgb 1 \
--learning_rate 0.01 \
--num_steps 60000 \
--save_dir "./save_dualseg_r101/cityscapes" \
--ohem 1 --ohem_thres 0.7 --ohem_keep 100000 \
--log_file "dual_seg_r50.log"
