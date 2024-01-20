#!/bin/bash

### general ###
scope=('LBAM_deoccnet_comb_7_5_1x1_v2');
random_seed=1234;
gpu_ids='0,1,2,3';
mode='train';
eval_epoch=1;

### dataset options ###
data_directory='/workspace/ssd1/datasets';
name_data='LFGAN';
x_res=600;
y_res=400;
uv_diameter_image=5;
uv_diameter=5;
data_output_option='2d_sub';
resize_scale=0.5;
num_workers=16;

### train options ###
model='LBAM_deoccnet_comb_7_5_1x1_v2';
views=25;
train_continue='off';
batch_size=16;
num_epoch=500;
alpha_size=-1;

## train optimizer options ##
learning_rate=5e-4;
scheduler='step';
scheduler_gamma=0.5;
scheduler_step=200;

## train loss options ##
loss_mode='Inpainting'
name_loss=('011'); # name loss: 0 or 1 for gan loss, perceptual loss, and style loss sequencially.

### log options ###
log_dir='./results/log';
checkpoint_dir='./results/checkpoints';
results_dir='./results/res_imgs';
name_metric='psnrssim';
log_iter=20;

for((i=0;i<1;i++)); do
    python -W ignore main.py \
        --scope ${scope[i]} \
        --random_seed $random_seed \
        --gpu_ids $gpu_ids \
        --mode $mode \
        --data_directory $data_directory \
        --name_data $name_data \
        --x_res $x_res \
        --y_res $y_res \
        --uv_diameter_image $uv_diameter_image \
        --uv_diameter $uv_diameter \
        --data_output_option $data_output_option \
        --resize_scale $resize_scale \
        --num_workers $num_workers \
        --model $model \
        --views $views \
        --train_continue $train_continue \
        --batch_size $batch_size \
        --num_epoch $num_epoch \
        --learning_rate $learning_rate \
        --scheduler $scheduler \
        --scheduler_gamma $scheduler_gamma \
        --scheduler_step $scheduler_step \
        --loss_mode $loss_mode \
        --name_loss ${name_loss[i]} \
        --tensorboard \
        --log_dir $log_dir \
        --checkpoint_dir $checkpoint_dir \
        --results_dir $results_dir \
        --name_metric $name_metric \
        --eval_epoch $eval_epoch \
        --log_iter $log_iter \
        --alpha_size $alpha_size;
done