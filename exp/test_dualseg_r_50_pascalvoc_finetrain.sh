#!/usr/bin/env bash

# train the net (suppose 8 gpus)
python eval.py --data_set pascalvoc \
--data_dir "./dataset/PASCALVOC2012/VOCdevkit/VOC2012" \
--input_size 321 \
--num_classes 21 \
--arch DualSeg_res50 \
--rgb 1 \
--restore_from "./save_dualseg_r50/pascalvoc/DualSeg_res50_final.pth" \
--output_dir "./dual_seg_r50/pascalvoc"
