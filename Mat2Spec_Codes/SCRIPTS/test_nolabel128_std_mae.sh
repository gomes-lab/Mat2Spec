CUDA_VISIBLE_DEVICES=0 python test_Mat2Spec.py \
--concat_comp '' \
--Mat2Spec-loss-type 'MAE' \
--label_scaling 'standardized' \
--data_src 'no_label_128' \
--trainset_subset_ratio 1.0 \
--check-point-path './TRAINED/model_Mat2Spec_binned_dos_128_standardized_MAE_trainsize1.0.chkpt' \
--Mat2Spec-label-dim 128
