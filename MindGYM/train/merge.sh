CUDA_VISIBLE_DEVICES=0,1,2,3 swift export \
    --adapters checkpoint_adapter_4_path \
    --merge_lora true \
    --output_dir checkpoint_all_path
