OPENAI_API_KEY=xxxxx \
CUDA_VISIBLE_DEVICES=1 MAX_PIXELS=1003520 swift eval \
  --model model_path \
  --ckpt_dir checkpoint_path \
  --port 8002 \
  --infer_backend pt \
  --eval_limit 500 \
  --eval_dataset MMMU_DEV_VAL \
  --eval_backend VLMEvalKit \
  --eval_num_proc 1
