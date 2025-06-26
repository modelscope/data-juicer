export VLLM_USE_MODELSCOPE=True
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
  --model EVAL_MODEL_PATH \
  --served-model-name qwen3-32b \
  --trust_remote_code \
  --tensor-parallel-size 4 \
  --port 8902 \
  --disable-log-stats
