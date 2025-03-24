OPENAI_API_KEY=xxxxx \
CUDA_VISIBLE_DEVICES=3 swift eval \
  --model model_path \
  --ckpt_dir checkpoint_path \
  --port 8004 \
  --infer_backend pt \
  --eval_limit 300 \
  --eval_dataset gpqa \
  --eval_backend Native \
  --eval_num_proc 1
