#!/bin/bash

##########################################################

# model path
PRETRAINED_MODEL_NAME_OR_PATH=$1
TRANSFORMER_PATH=$2
LORA_PATH=$3

# inferance config
IMAGE_SIZE=$4
PROMPT_INFO_PATH=$5
GPU_NUM=$6
BATCH_SIZE=$7
MIXED_PRECISION=$8
VIDEO_NUM_PER_PROMPT=$9
SEED=${10}

# saving config
OUTPUT_VIDEO_DIR=${11}

##########################################################



# run
for (( i = 0; i < GPU_NUM; i++ )); do
{
    CUDA_VISIBLE_DEVICES=$i python infer_lora.py \
      --prompt_info_path=$PROMPT_INFO_PATH \
      --config_path "config/easyanimate_video_motion_module_v1.yaml" \
      --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME_OR_PATH \
      --transformer_path=$TRANSFORMER_PATH \
      --lora_path=$LORA_PATH \
      --image_size=$IMAGE_SIZE \
      --chunks_num=$GPU_NUM \
      --chunk_id=$i \
      --batch_size=$BATCH_SIZE \
      --video_num_per_prompt=$VIDEO_NUM_PER_PROMPT \
      --mixed_precision=$MIXED_PRECISION \
      --save_path=$OUTPUT_VIDEO_DIR \
      --seed=$SEED
} &   
done

wait

