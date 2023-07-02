#!/bin/bash

# Change home and conda directory to own path
home=/home/paul
conda=/home/paul/miniconda3/bin/activate
log=${home}/world-SSMs/logdir

#export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
#cd ${home}/jpl-source-seeking/DRL-robot-navigation/catkin_ws
#source devel_isolated/setup.bash

source ${conda} wssm

python3 ${home}/world-SSMs/main.py \
    --configs atari medium \
    --logdir ${log}/run0 \
    --replay_size 1000000.0 \
    --replay_online True \
    --jax.platform gpu \
    --jax.policy_devices 0 \
    --jax.train_devices 0 \
    --jax.prealloc False \
    --run.script train \
    --run.steps 1e10 \
    --run.eval_every 1e6 \
    --run.train_ratio 32 \
    --batch_size 16 \
    --encoder.mlp_keys 'vector' \
    --decoder.mlp_keys 'vector' \
    --encoder.mlp_units 512 \
    --decoder.mlp_units 512 \

    # Some default settings for reference
    #--run.steps 1e10 \
    #--run.eval_every 1e6 \
    #--encoder.mlp_keys: '$^' \
    #--decoder.mlp_keys: '$^' \
    #--encoder.cnn_keys: 'image' \
    #--decoder.cnn_keys: 'image'
