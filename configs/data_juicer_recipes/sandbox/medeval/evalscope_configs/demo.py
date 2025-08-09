import argparse

from evalscope import TaskConfig, run_task

parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", type=str, default="outputs")
args = parser.parse_args()

task_cfg = TaskConfig(
    model="qwen25-1.5b",
    api_url="http://127.0.0.1:8901/v1/chat/completions",
    api_key="EMPTY",  # pragma: allowlist secret
    eval_type="service",
    datasets=["gsm8k", "arc"],
    work_dir=args.work_dir,
    limit=20,
)

run_task(task_cfg=task_cfg)
