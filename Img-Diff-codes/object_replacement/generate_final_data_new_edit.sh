#!/bin/bash

CUDA_VISIBLE_DEVICES="0" python generate_final_data_new_edit.py \
    --inp2p_bbox_json_path ./bbox_new_edit_09_098_only_target_025sam_0.json \
    --llava_path llava-v1.6-vicuna-7b \
    --clip_path clip-vit-base-patch32 \
    --blip_path blip-itm-large-coco \
    --img_dir ./prompt-to-prompt-with-sdxl/output \
    --output_img_dir ./filtered_new_edit_data_9_98_only_target_025sam \
    --qa_turns 5 \
    --output_file ./generate_final_data_only_target_9_98_025sam_0.json