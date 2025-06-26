export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=2,3 python tools/sandbox_starter.py --config medeval/yaml/start.yaml
