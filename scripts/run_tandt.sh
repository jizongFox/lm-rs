#!/bin/bash

tandt_scenes=("truck" "train")

# Scheduler Settings. Milestones and Values
BATCHSIZE_VALUES="16,32"
BATCH_BREAKS="50"
SAMPLE_BREAKS="50"
SAMPLE_VALS="32,32"
CG_BREAKS="50"
CG_VALS="5,8"

LOSS_FN="mse" # "mse" or "mse_approx_ssim"
SSIM_WEIGHT=0.0
if [ "$LOSS_FN" = "mse_approx_ssim" ]; then
    SSIM_WEIGHT=0.2
fi


for scene in "${tandt_scenes[@]}"; do
    SOURCE_PATH="datasets/tandt/${scene}"
    MODEL_PATH="output/${scene}"
    mkdir -p $MODEL_PATH
    python train.py \
            --auto_lr --max_lr 0.2 \
            --regularizer=0.1 --source_path $SOURCE_PATH --model_path $MODEL_PATH \
            --eval --iterations=130 --sh_degree=0 --log_freq=10 --log_image_freq=100 \
            --sampling_distribution uniform --camera_sampler clusterdir \
            --cgiter_values $CG_VALS --cgiter_breakpoints $CG_BREAKS \
            --batchsize_values $BATCHSIZE_VALUES --batchsize_breakpoints $BATCH_BREAKS \
            --samplesize_values $SAMPLE_VALS --samplesize_breakpoints $SAMPLE_BREAKS \
            --loss_fn $LOSS_FN --ssim_weight $SSIM_WEIGHT
done 