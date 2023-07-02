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

python ${home}/world-SSMs/main.py \
    --configs crafter medium \
    --logdir ${log}/run_crafter_0 \
    --replay_size 1000000.0 \
    --replay_online True \
    --jax.platform gpu \
    --jax.policy_devices 0 \
    --jax.train_devices 0 \
    --jax.prealloc False \
    --run.script train_save \
    --run.steps 1e10 \
    --run.eval_every 1e6 \
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
