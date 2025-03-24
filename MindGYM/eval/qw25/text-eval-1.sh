OPENAI_API_KEY=xxxxx \
CUDA_VISIBLE_DEVICES=2 swift eval \
  --model model_path \
  --ckpt_dir checkpoint_path \
  --port 8003 \
  --infer_backend pt \
  --eval_dataset aime24 gsm8k math_500 trivia_qa \
  --eval_backend Native \
  --eval_num_proc 1
