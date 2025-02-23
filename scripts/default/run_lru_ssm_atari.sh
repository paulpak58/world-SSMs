#!/bin/bash

# Change home and conda directory to own path
home=/home/paul
conda=/home/paul/miniconda3/bin/activate
log=${home}/world-SSMs/logdir

source ${conda} wssm


####################### TEST LOG ###########################
# logdir:
#       - run_lru_ssm_atari: small lru [CUDA 0] 
#############################################################
python ${home}/world-SSMs/ssm_main.py \
    --configs atari small \
    --logdir ${log}/run_lru_ssm_atari \
    --replay_size 1e6 \
    --replay_online False \
    --jax.platform gpu \
    --jax.policy_devices 0 \
    --jax.train_devices 0 \
    --jax.prealloc False \
    --run.script train \
    --run.steps 1e6 \
    --run.eval_every 1e5 \
    --batch_size 4 \
    --imag_horizon 16 \
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
