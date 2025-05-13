python data_construction_pipeline/detect_character_attributes.py \
--qwen2_5_vl_model_path Qwen/Qwen2.5-VL-7B-Instruct \
--data_json_path ./playground/data_construction/docci_character_location.json \
--image_folder ./playground/docci/images \
--output_json ./playground/data_construction/docci_character_attributes.json \
--gpu_nums 1



python data_construction_pipeline/detect_character_attributes.py \
--qwen2_5_vl_model_path Qwen/Qwen2.5-VL-7B-Instruct \
--data_json_path ./playground/data_construction/flickr30k_character_location.json \
--image_folder ./playground/localized_narratives/flickr30k/flickr30k_images \
--output_json ./playground/data_construction/flickr30k_character_attributes.json \
--gpu_nums 1



python data_construction_pipeline/detect_character_attributes.py \
--qwen2_5_vl_model_path Qwen/Qwen2.5-VL-7B-Instruct \
--data_json_path ./playground/data_construction/coco_character_location.json \
--image_folder ./playground/localized_narratives/coco/train2017 \
--output_json ./playground/data_construction/coco_character_attributes.json \
--gpu_nums 1