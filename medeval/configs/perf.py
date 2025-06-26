import argparse

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='outputs')
args = parser.parse_args()

task_cfg = Arguments(parallel=[1, 100],
                     number=[10, 200],
                     model='qwen25-1.5b',
                     url='http://127.0.0.1:8901/v1/chat/completions',
                     api='openai',
                     dataset='openqa',
                     temperature=0.9,
                     max_tokens=1024,
                     min_prompt_length=10,
                     max_prompt_length=4096,
                     tokenizer_path=INFER_MODEL_PATH,
                     extra_args={'ignore_eos': True},
                     outputs_dir=args.work_dir)
results = run_perf_benchmark(task_cfg)
