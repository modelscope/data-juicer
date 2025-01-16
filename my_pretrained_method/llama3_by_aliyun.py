import transformers
import torch
import transformers
import torch
from modelscope import snapshot_download
import os
os.environ['TF_ENABLE_ONEDNN_OPT'] = '2'

# model_id = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct")


model_id = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct",
                  cache_dir="/mnt/zt_pt_model/Meta-Llama-3.1-8B-Instruct")

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# pipeline.eval()

outputs = pipeline(
    messages,
    temperature = 0.1,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])

