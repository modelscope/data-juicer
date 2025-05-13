python data_construction_pipeline/filter_feature.py \
--qwen2_5_model_path Qwen/Qwen2.5-VL-7B-Instruct \
--polished_prompt_data_json_path ./playground/data_construction/docci_polished_prompt.json \
--character_attributes_data_json_path ./playground/data_construction/docci_character_attributes.json \
--character_locations_data_json_path ./playground/data_construction/docci_bbox_to_location.json \
--scene_attributes_data_json_path ./playground/data_construction/docci_sence_attributes.json \
--image_folder ./playground/docci/images \
--output_json ./playground/data_construction/docci_final_polished_prompt.json \
--gpu_nums 1


python data_construction_pipeline/filter_feature.py \
--qwen2_5_model_path Qwen/Qwen2.5-VL-7B-Instruct \
--polished_prompt_data_json_path ./playground/data_construction/flickr30k_polished_prompt.json \
--character_attributes_data_json_path ./playground/data_construction/flickr30k_character_attributes.json \
--character_locations_data_json_path ./playground/data_construction/flickr30k_bbox_to_location.json \
--scene_attributes_data_json_path ./playground/data_construction/flickr30k_sence_attributes.json \
--image_folder ./playground/localized_narratives/flickr30k/flickr30k_images \
--output_json ./playground/data_construction/flickr30k_final_polished_prompt.json \
--gpu_nums 1


python data_construction_pipeline/filter_feature.py \
--qwen2_5_model_path Qwen/Qwen2.5-VL-7B-Instruct \
--polished_prompt_data_json_path ./playground/data_construction/coco_polished_prompt.json \
--character_attributes_data_json_path ./playground/data_construction/coco_character_attributes.json \
--character_locations_data_json_path ./playground/data_construction/coco_bbox_to_location.json \
--scene_attributes_data_json_path ./playground/data_construction/coco_sence_attributes.json \
--image_folder ./playground/localized_narratives/coco/train2017 \
--output_json ./playground/data_construction/coco_final_polished_prompt.json \
--gpu_nums 1