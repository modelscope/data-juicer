OPENAI_API_KEY=xxxxx \
CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=1003520 swift eval \
  --model model_path \
  --ckpt_dir checkpoint_path \
  --port 8001 \
  --infer_backend pt \
  --eval_dataset Mathvista_MINI Mathvision_MINI MMStar \
  --eval_backend VLMEvalKit \
  --eval_num_proc 1
