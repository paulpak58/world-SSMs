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
    --configs crafter xlarge  \
    --logdir ${log}/run_crafter_large \
    --replay_size 1000000.0 \
    --jax.platform gpu \
    --jax.prealloc False \
    --run.script train_save \
    --run.steps 1e7 \
    --run.eval_every 1e7