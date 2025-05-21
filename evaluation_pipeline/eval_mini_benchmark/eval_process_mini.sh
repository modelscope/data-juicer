python evaluation_pipeline/eval_mini_benchmark/eval_process_mini.py \
--qwen2_5_vl_model_path Qwen/Qwen2.5-VL-7B-Instruct \
--yoloe_model_path yoloe-11l-seg.pt \
--blip_model_path Salesforce/blip-itm-large-flickr \
--ann_json_path ./DetailMaster_Dataset/DetailMaster_mini_benchmark.json \
--image_folder path_to_your_generated_images \
--image_info_json path_to_the_annotations_of_your_generated_images \
--output_log_dir ./playground/evaluation \
--output_name_prefix your_model_s_name
