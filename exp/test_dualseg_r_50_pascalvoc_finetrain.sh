#!/usr/bin/env bash

# train the net (suppose 8 gpus)
python3 eval_pascalvoc.py --data_set pascalvoc \
--data_dir "/home/anand17218/GALD-DGCNet/dataset/PASCALVOC2012/VOCdevkit/VOC2012" \
--num_classes 21 \
--input_size 384 \
--arch DualSeg_res50 \
--rgb 1 \
--whole True \
--restore_from "./save_dualseg_r50/pascalvoc/DualSeg_res50_500epoch.pth" \
