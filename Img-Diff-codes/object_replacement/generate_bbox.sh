CUDA_VISIBLE_DEVICES="0" python generate_bbox.py \
    --vit_path "clip-vit-base-patch32" \
    --blip_path "blip-itm-large-coco" \
    --fastsam_path "FastSAM-x.pt" \
    --json_path "filtered_file_new_edit_09_098_3.json" \
    --output_file "bbox_file_3.json"

