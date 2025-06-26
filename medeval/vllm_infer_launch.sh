export VLLM_USE_MODELSCOPE=True
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model INFER_MODEL_PATH \
  --served-model-name qwen25-1.5b \
  --trust_remote_code \
  --tensor-parallel-size 2 \
  --port 8901 \
  --disable-log-stats
