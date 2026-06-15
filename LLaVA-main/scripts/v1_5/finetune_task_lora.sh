#!/bin/bash

# 设置显卡，不使用 CUDA_VISIBLE_DEVICES，使用 deepspeed --include 设置显卡 ID
include=localhost:0,1 # 使用 GPU 0 和 1

# 模型和数据路径
model_name_or_path=/home/dongchenxi/shiyan/llava/LLaVA-main/llava-v1.5-7b
data_path=/home/dongchenxi/shiyan/llava/LLaVA-main/playground/data/train_mix_pure.json
image_folder=/home/dongchenxi/shiyan/llava/LLaVA-main/playground/data/test/images

deepspeed --include $include llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --mm_projector_lr 1e-4 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $model_name_or_path \
    --version v1 \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower /home/dongchenxi/shiyan/llava/LLaVA-main/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-task-lora \
    --num_train_epochs 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 3e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none
