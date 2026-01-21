#!/bin/bash



# More information about the tool can be found in https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html

SOURCE_PATH="datasets/tandt/truck"
MODEL_PATH="output/profiler"

mkdir -p $MODEL_PATH

$NCU_PATH/ncu --target-processes application-only --kernel-name regex:".*JTv_renderCUDA.*|.*forwardDiffRender.*|.*Diag.*" --set full --page details -f -o ${MODEL_PATH}/profile_result python train.py \
        --disable_scheds --batch_size=1 --fixed_lr=0.03 --cg_iter=1 --regularizer=1e-2 --N_sample_per_tile=256 \
        --source_path $SOURCE_PATH --model_path $MODEL_PATH --log_freq=-1 \
        --eval --sh_degree=0 --iterations=3 --camera_sampler=clusterdir
