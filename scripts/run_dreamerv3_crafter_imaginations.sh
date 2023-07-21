#!/bin/bash

# Change home and conda directory to own path
home=/home/paul
conda=/home/paul/miniconda3/bin/activate
log=${home}/world-SSMs/logdir

source ${conda} wssm



####################### TEST LOG ###########################
# logdir:
#   - run_crafter_large_imag_128: imagination horizon = 128, gpu devices = 0
#   - run_crafter_large_imag_256: imagination horizon = 256, gpu devices = 1
#   - run_crafter_large_imag_512: imagination horizon = 512, gpu devices = 2
#
#   - run_crafter_small_imag_128: imagination horizon = 128, gpu devices = 0
#   - run_crafter_small_imag_256: imagination horizon = 256, gpu devices = 1
#   - run_crafter_small_imag_512: imagination horizon = 512, gpu devices = 2

#############################################################
python ${home}/world-SSMs/main.py \
    --configs crafter small \
    --logdir ${log}/run_crafter_small_imag_32 \
    --replay_size 1e6 \
    --replay_online False \
    --jax.platform gpu \
    --jax.policy_devices 1 \
    --jax.train_devices 1 \
    --jax.prealloc False \
    --run.script train \
    --run.steps 1e6 \
    --run.eval_every 1e6 \
    --batch_size 16 \
    --imag_horizon 32
    # --run.train_ratio 32 \


    #--encoder.mlp_keys 'vector' \
    #--decoder.mlp_keys 'vector' \
    #--encoder.mlp_units 512 \
    #--decoder.mlp_units 512 \

    # Some default settings for reference
    #--run.steps 1e10 \
    #--run.eval_every 1e6 \
    #--encoder.mlp_keys: '$^' \
    #--decoder.mlp_keys: '$^' \
    #--encoder.cnn_keys: 'image' \
    #--decoder.cnn_keys: 'image'
