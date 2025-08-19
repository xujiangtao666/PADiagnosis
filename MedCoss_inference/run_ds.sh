#!/bin/bash

gpu_id=1

task_id='3D_RICORD'

reload_from_pretrained=True
pretrained_path='/ifs/home/wushangqian/MAE/Pretrained_Model/MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05/checkpoint-0.pth'
exp_name='MedCoSS_Report_Xray_CT_MR_Path_Buffer0.05'

data_path='/ifs/data/wushangqian/Continual_learning/3D/RICORD/'
epoch=200

lr=0.00001

seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dirss
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/RICORD/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='64,192,192' \
--batch_size=8 \
--num_gpus=1 \
--num_epochs=$epoch \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=2 \
--num_workers=10 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed

seed=10
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/RICORD/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='64,192,192' \
--batch_size=8 \
--num_gpus=1 \
--num_epochs=$epoch \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=2 \
--num_workers=10 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed


seed=100
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir='snapshots/downstream/dim_3/'$path_id
mkdir $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_3/RICORD/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='64,192,192' \
--batch_size=8 \
--num_gpus=1 \
--num_epochs=$epoch \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=2 \
--num_workers=10 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed