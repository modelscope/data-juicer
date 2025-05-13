python data_construction_pipeline/detect_character_locations.py \
--qwen2_5_vl_model_path Qwen/Qwen2.5-VL-7B-Instruct \
--yoloe_model_path yoloe-11l-seg.pt \
--data_json_path ./playground/data_construction/docci_main_character_filtered.json \
--image_folder ./playground/docci/images \
--output_json ./playground/data_construction/docci_character_location.json \
--blip_model_path Salesforce/blip-itm-large-flickr \
--gpu_nums 1


python data_construction_pipeline/detect_character_locations.py \
--qwen2_5_vl_model_path Qwen/Qwen2.5-VL-7B-Instruct \
--yoloe_model_path yoloe-11l-seg.pt \
--data_json_path ./playground/data_construction/flickr30k_main_character_filtered.json \
--image_folder ./playground/localized_narratives/flickr30k/flickr30k_images \
--output_json ./playground/data_construction/flickr30k_character_location.json \
--blip_model_path Salesforce/blip-itm-large-flickr \
--gpu_nums 1


python data_construction_pipeline/detect_character_locations.py \
--qwen2_5_vl_model_path Qwen/Qwen2.5-VL-7B-Instruct \
--yoloe_model_path yoloe-11l-seg.pt \
--data_json_path ./playground/data_construction/coco_main_character_filtered.json \
--image_folder ./playground/localized_narratives/coco/train2017 \
--output_json ./playground/data_construction/coco_character_location.json \
--blip_model_path Salesforce/blip-itm-large-flickr \
--gpu_nums 1
