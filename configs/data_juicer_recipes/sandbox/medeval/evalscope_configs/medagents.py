import argparse

from evalscope import TaskConfig, run_task

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='outputs')
args = parser.parse_args()

task_cfg = TaskConfig(
    model='qwen25-1.5b',
    api_url='http://127.0.0.1:8901/v1/chat/completions',
    api_key='EMPTY',  # pragma: allowlist secret
    eval_type='service',
    datasets=['general_mcq'],
    dataset_args={
        'general_mcq': {
            'local_path':
            'medeval/data/med_data_sub/medagents',
            'subset_list': [
                'afrimedqa', 'medbullets', 'medexqa', 'medmcqa',
                'medqa_5options', 'medqa', 'medxpertqa-r', 'medxpertqa-u',
                'mmlu', 'mmlu-pro', 'pubmedqa'
            ],
            'prompt_template':
            'Please answer this medical question and select the correct answer\n{query}',
            'query_template':
            'Question: {question}\n{choices}\nAnswer: {answer}\n\n',
        }
    },
    work_dir=args.work_dir,
    limit=20,
)

run_task(task_cfg=task_cfg)
