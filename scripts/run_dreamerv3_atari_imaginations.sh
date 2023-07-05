#!/bin/bash

# Change home and conda directory to own path
home=/home/paul
conda=/home/paul/miniconda3/bin/activate
log=${home}/world-SSMs/logdir

source ${conda} wssm



####################### TEST LOG ###########################
# logdir:
#   - run_atari_medium_imag_64: imagination horizon = 64
#   - run_atari_medium_imag_128: imagination horizon = 128
#   - run_atari_medium_imag_256: imagination horizon = 256

#############################################################
python ${home}/world-SSMs/main.py \
    --configs atari medium \
    --logdir ${log}/run_atari_medium_imag_64 \
    --replay_size 1000000.0 \
    --replay_online False \
    --jax.platform gpu \
    --jax.policy_devices 0 \
    --jax.train_devices 0 \
    --jax.prealloc False \
    --run.script train \
    --run.steps 1e7 \
    --run.eval_every 1e5 \
    --batch_size 16 \
    --imag_horizon 64
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
