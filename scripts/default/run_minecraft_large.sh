#!/bin/bash

# Change home and conda directory to own path
home=/home/paul
conda=/home/paul/miniconda3/bin/activate
log=${home}/world-SSMs/logdir

#export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
#source devel_isolated/setup.bash

source ${conda} wssm

export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
export PATH="/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/:$PATH"

xvfb-run python3 ${home}/world-SSMs/main.py \
    --configs minecraft xlarge  \
    --logdir ${log}/run_minecraft_large \
    --replay_size 1000000.0 \
    --jax.platform gpu \
    --jax.prealloc False \
    --run.script train_save \
    --run.steps 1e7 \
    --run.eval_every 1e7
