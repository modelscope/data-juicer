python -m torch.distributed.launch --nproc_per_node=4 gen_new_data_ddp.py \
    --model_path stable-diffusion-xl-base-1.0 \
    --json_path ./gen_vg.json \
    --output_path ./output_vg_ddp