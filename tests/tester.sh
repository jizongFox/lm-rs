#!/bin/bash

SOURCE_PATH="datasets/nerf_synthetic/lego"
IMAGES="images"

LOSS_FN="mse" 
SSIM_WEIGHT=0.0

TOTAL_BATCH=4


# PATHS 
ADAM_PATH="tests/adam_ckpt_iter1000/"
JACOBIAN_PATH="tests/GT_Jacobians/"


# Run Adam, create a ckpt to start from 
# If Adam path does not exist
if [ ! -d "$ADAM_PATH" ] && [ ! -L "$ADAM_PATH" ]; then 
    ITERATIONS=1000
    OPTIMIZER=adam

    SSIMW=0.2
    POS_LR_INIT=0.00016
    POS_LR_FINAL=0.0000016
    FEATURE_LR=0.0025
    OPACITY_LR=0.05
    SCALING_LR=0.005
    ROTATION_LR=0.001

    echo $ADAM_PATH
    mkdir -p $ADAM_PATH
    python train.py -r=8 --lambda_dssim $SSIMW --source_path $SOURCE_PATH --model_path $ADAM_PATH \
    --sh_degree=0 --optimization_method $OPTIMIZER --eval --images $IMAGES --iterations $ITERATIONS --loss_fn $LOSS_FN --log_freq=100 --log_image_freq=10000 \
    --checkpoint_iterations 1000 --position_lr_init $POS_LR_INIT --position_lr_final $POS_LR_FINAL --feature_lr $FEATURE_LR \
    --opacity_lr $OPACITY_LR --scaling_lr $SCALING_LR --rotation_lr $ROTATION_LR > ${ADAM_PATH}/log.txt 2>${ADAM_PATH}/error.txt
fi
# End of Adam ckpt

# Load saved Gaussians, save the Jacobians using batch size TOTAL_BATCH.
if [ ! -d "$JACOBIAN_PATH" ] && [ ! -L "$JACOBIAN_PATH" ]; then 
    echo $JACOBIAN_PATH
    mkdir -p $JACOBIAN_PATH
    python train.py -r=8 --images $IMAGES --camera_sampler sequential --source_path $SOURCE_PATH --model_path $JACOBIAN_PATH \
            --eval --optimization_method=cg-gpu --kernel=-1 --loss_fn $LOSS_FN --iterations=1001  --sh_degree=0 \
            --disable_scheds --batch_size $TOTAL_BATCH \
            --start_checkpoint=tests/adam_ckpt_iter1000/chkpnt1000.pth >${JACOBIAN_PATH}/log.txt 2>${JACOBIAN_PATH}/error.txt
fi
# #### End of Jacobian dumping

# # Dump the results of custom CUDA implementation
MODEL_PATH="tests/gauss-newton-ckpt/"
mkdir -p $MODEL_PATH
python train.py --images $IMAGES --camera_sampler sequential --source_path $SOURCE_PATH --model_path $MODEL_PATH \
        --eval --optimization_method=cg-gpu --kernel=1 --loss_fn $LOSS_FN --iterations=1001 --sh_degree=0 \
        --ssim_weight $SSIM_WEIGHT --sampling_distribution uniform -r=8 \
        --disable_scheds --batch_size $TOTAL_BATCH --return_matvec_kernels --save=True \
        --start_checkpoint=tests/adam_ckpt_iter1000/chkpnt1000.pth >${MODEL_PATH}/log.txt 2>${MODEL_PATH}/error.txt

#### End of dumping CUDA results

# Check if GT values match with CUDA
python tests/test_cuda.py --batch_size $TOTAL_BATCH