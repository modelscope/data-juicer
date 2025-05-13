python data_construction_pipeline/detect_main_character.py \
--qwen2_5_vl_model_path Qwen/Qwen2.5-VL-7B-Instruct \
--data_json_path ./playground/docci/docci_descriptions.json \
--image_folder ./playground/docci/images \
--output_json ./playground/data_construction/docci_main_character.json \
--dataset_target docci \
--gpu_nums 1


python data_construction_pipeline/detect_main_character.py \
--qwen2_5_vl_model_path Qwen/Qwen2.5-VL-7B-Instruct \
--data_json_path ./playground/localized_narratives/flickr30k/flickr30k_train_localized_narratives.json \
--image_folder ./playground/localized_narratives/flickr30k/flickr30k_images \
--output_json ./playground/data_construction/flickr30k_main_character.json \
--dataset_target ln_flickr \
--gpu_nums 1


python data_construction_pipeline/detect_main_character.py \
--qwen2_5_vl_model_path Qwen/Qwen2.5-VL-7B-Instruct \
--data_json_path ./playground/localized_narratives/coco/coco_train_captions.json \
--image_folder ./playground/localized_narratives/coco/train2017 \
--output_json ./playground/data_construction/coco_main_character.json \
--dataset_target ln_coco \
--gpu_nums 1