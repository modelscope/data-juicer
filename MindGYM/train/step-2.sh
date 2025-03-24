nproc_per_node=4

MASTER_PORT=29505 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model_type qwen2_5_vl \
    --model model_path \
    --resume_from_checkpoint checkpoint_adapter_1_path \
    --resume_only_model \
    --dataset data_2_path \
    --freeze_llm False \
    --freeze_vit True \
    --freeze_aligner True \
    --split_dataset_ratio 0.03 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 8 / $nproc_per_node) \
    --eval_steps 10 \
    --save_steps 60 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir adapter_2_path \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --seed 42 \
    --deepspeed zero3
