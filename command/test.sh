#!/bin/bash
### general ###
scope=('LBAM_deoccnet_7_5_1x1_v2');

random_seed=1234;
gpu_ids='3';
mode='test';
eval_epoch=(500 400 300);

### dataset options ###
data_directory='/workspace/ssd1/datasets';
name_data=('DeOccNet' 'DeOccNet' 'DeOccNet' 'DeOccNet' 'test_Stanford_Lytro_16' 'test_EPFL_10' 'Dense_Quant' 'Dense_Quant_Double');
save_data=(1 1 1 1 1 1 0 0);
x_res=(600 600 600 512 600 600 600 600);
y_res=(400 400 400 512 400 400 400 400);
uv_diameter_image=(5 5 5 5 9 9 5 5);
uv_diameter=5;
uv_dilation=1;
data_output_option='2d_sub';
resize_scale=0.5;
# resize_scale=1.0;
num_workers=16;

### train options ###
model=('LBAM_deoccnet_comb_DF_v2' 'LBAM_deoccnet_comb_7_5_1x1_v12' 'LBAM_deoccnet_comb_7_5_1x1_v12');
views=25;
train_continue='on';
batch_size=1;
num_epoch=250;
alpha_size=3
specific_dir=('Realscenes' 'Synscenes'  'CD' 'Synscenes9' '' '' 'occluded' 'occluded');

## train optimizer options ##
learning_rate=1e-3;

## train loss options ##
loss_mode='Inpainting'
name_loss=('111');

### log options ###
log_dir='./results/log';
checkpoint_dir='./results/checkpoints';
results_dir='./results/res_imgs';
name_metric='psnrssim';
log_iter=100;
for((k=0;k<1;k++)); do # eval epoch
    for((j=0;j<1;j++)); do # scope, model
        for((i=0;i<8;i++)); do # dataset
            python -W ignore main.py \
                --scope ${scope[j]} \
                --random_seed $random_seed \
                --gpu_ids $gpu_ids \
                --mode $mode \
                --data_directory $data_directory \
                --specific_dir=${specific_dir[i]} \
                --name_data ${name_data[i]} \
                --save_data ${save_data[i]} \
                --x_res ${x_res[i]} \
                --y_res ${y_res[i]} \
                --uv_diameter_image ${uv_diameter_image[i]} \
                --uv_diameter $uv_diameter \
                --uv_dilation $uv_dilation \
                --data_output_option $data_output_option \
                --resize_scale $resize_scale \
                --num_workers $num_workers \
                --model ${model[j]} \
                --views $views \
                --train_continue $train_continue \
                --batch_size $batch_size \
                --num_epoch $num_epoch \
                --learning_rate $learning_rate \
                --loss_mode $loss_mode \
                --name_loss ${name_loss} \
                --tensorboard \
                --log_dir $log_dir \
                --checkpoint_dir $checkpoint_dir \
                --results_dir $results_dir \
                --name_metric $name_metric \
                --eval_epoch ${eval_epoch[k]} \
                --log_iter $log_iter \
                --alpha_size $alpha_size;
        done
    done
done