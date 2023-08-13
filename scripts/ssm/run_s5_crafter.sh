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
    --configs crafter small \
    --logdir ${log}/run_s5_crafter \
    --jax.platform gpu \
    --jax.policy_devices 0 \
    --jax.train_devices 0 \
    --jax.prealloc False \
    --run.script train \
    --run.steps 1e6 \
    --run.eval_every 1e6 \
    --batch_size 16 \
    --imag_horizon 15 \
    --ssm s5 \
    --rssm.initial zeros

