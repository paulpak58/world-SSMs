#!/bin/bash

# Change home and conda directory to own path
home=/home/paul
conda=/home/paul/miniconda3/bin/activate
log=${home}/world-SSMs/ssm_logdir

source ${conda} wssm



####################### TEST LOG ###########################
# logdir:

#############################################################
python ${home}/world-SSMs/main.py \
    --configs crafter medium \
    --logdir ${log}/run_s5_crafter \
    --replay_size 1e6 \
    --replay_online False \
    --jax.platform gpu \
    --jax.policy_devices 2 \
    --jax.train_devices 2 \
    --jax.prealloc False \
    --run.script train \
    --run.steps 1e6 \
    --run.eval_every 1e6 \
    --batch_size 16 \
    --imag_horizon 512 \
    --ssm s5
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
