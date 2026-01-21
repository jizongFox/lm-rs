# More information about the tool can be found in https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html

SOURCE_PATH="datasets/tandt/truck"
MODEL_PATH="output/sanitizer"

mkdir -p $MODEL_PATH
compute-sanitizer --tool memcheck python train.py --disable_scheds \
        --batch_size=1 --fixed_lr=0.03 --cg_iter=8 --regularizer=1e-2 --N_sample_per_tile=32 \
        --source_path $SOURCE_PATH --model_path $MODEL_PATH --log_freq=-1 \
        --eval --sh_degree=0 --iterations=3 --camera_sampler=clusterdir
