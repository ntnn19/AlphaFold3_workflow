#!/bin/bash

# Determine number of GPUs on the node we ended up on.
JOB_LIST=$1
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo $NUM_GPUS
< $JOB_LIST parallel -j $NUM_GPUS 'eval CUDA_VISIBLE_DEVICES=$(({%} - 1)) {}'
