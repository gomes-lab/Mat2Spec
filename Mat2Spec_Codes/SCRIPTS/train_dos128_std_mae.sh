#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train_Mat2Spec.py \
--concat_comp '' \
--Mat2Spec-loss-type 'MAE' \
--label_scaling 'standardized' \
--data_src 'binned_dos_128' \
--trainset_subset_ratio 1.0 \
--train \
--Mat2Spec-label-dim 128 \

