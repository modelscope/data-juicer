CUDA_VISIBLE_DEVICES="0" python generate_inpaint.py \
    --out_path ./inpaint_all_mscoco \
    --vit_path clip-vit-base-patch32 \
    --blip_path blip-itm-large-coco \
    --fastsam_path FastSAM-x.pt \
    --sd_model_path sdxl-turbo \
    --mllm_path llava-v1.6-vicuna-7b \
    --json_path filtered_file_new_edit_all_0.json \
    --output_file inpaint_mllm_all_mscoco_0.json \
    --split_name 0

