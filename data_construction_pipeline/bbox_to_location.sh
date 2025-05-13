python data_construction_pipeline/bbox_to_location.py \
--character_bbox_data_json_path ./playground/data_construction/docci_character_location.json \
--image_folder ./playground/docci/images \
--output_json ./playground/data_construction/docci_bbox_to_location.json


python data_construction_pipeline/bbox_to_location.py \
--character_bbox_data_json_path ./playground/data_construction/flickr30k_character_location.json \
--image_folder ./playground/localized_narratives/flickr30k/flickr30k_images \
--output_json ./playground/data_construction/flickr30k_bbox_to_location.json


python data_construction_pipeline/bbox_to_location.py \
--character_bbox_data_json_path ./playground/data_construction/coco_character_location.json \
--image_folder ./playground/localized_narratives/coco/train2017 \
--output_json ./playground/data_construction/coco_bbox_to_location.json