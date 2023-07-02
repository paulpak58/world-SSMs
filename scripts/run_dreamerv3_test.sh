#!/bin/bash

# Change home and conda directory to own path
home=/home/paul
conda=/home/paul/miniconda3/bin/activate
log=${home}/world-SSMs/logdir

source ${conda} wssm

python ${home}/world-SSMs/main.py \
    --configs atari small \
    --logdir ${log}/run_atari_test \
    --replay_size 1000000.0 \
    --replay_online False \
    --jax.platform gpu \
    --jax.policy_devices 0 \
    --jax.train_devices 1 \
    --jax.prealloc False \
    --run.script train \
    --run.steps 1e5 \
    --run.eval_every 1e5 \
    --batch_size 16 \
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
