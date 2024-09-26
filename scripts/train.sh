#!/bin/bash

cd ../src && 

CUDA_VISIBLE_DEVICES=0 nohup python train.py \
   --model_name=PickCube-v0_model \
   --num_traj=500 --n_iters=1_600_000 \
   --context_length=60 --model_type=s+a+cot \
   --task=PickCube-v0 --key_state_coeff=0.1 \
   --key_state_loss=0 --key_states=abcdef \
   --init_lr=5e-4 --num_workers=20 --seed=0 \
   --modify_dataset=True &
