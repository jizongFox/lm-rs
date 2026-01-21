#!/bin/bash

# Check if CUDA version is correct. It should be 11.8
nvcc --version
which nvcc

# Optional Cleaning
cd submodules/diff-gaussian-rasterization/
python setup.py clean --all
cd ../..

python -m pip install -e submodules/diff-gaussian-rasterization