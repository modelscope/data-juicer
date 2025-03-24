from modelscope import snapshot_download
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
import re
import json
import numpy as np
import base64
import argparse
import random
import string
import time
import torch
import config_img_cn
from tqdm import tqdm
import multiprocessing
import os
from torch import cuda
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


model_dir  = snapshot_download('Qwen/Qwen2.5-VL-72B-Instruct')


def extract_json(string):
    # Define a regular expression pattern to match JSON data in ```json``` tags
    pattern = r'```json\s*(\{.*?\})\s*```'

    # Use the re.DOTALL flag to match multiple lines
    match = re.search(pattern, string, re.DOTALL)
    
    if match:
        json_str = match.group(1) # Extract the JSON part
        try:
            json_data = json.loads(json_str) # Parse the JSON string into a Python dictionary
            return json_data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
    else:
        print("No valid JSON data found")
        return None


def retry_on_error(func, max_retries=5, delay=1):
    """
    Decorator function with retry mechanism
    :param func: function to be retried
    :param max_retries: maximum number of retries
    :param delay: delay time before each retry (seconds)
    :return: function execution result
    """
    def wrapper(*args, **kwargs):
        retries = 0
        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                print(f"Error: {e}, retry {retries}/{max_retries}...")
                if retries >= max_retries:
                    raise # retries are exhausted, throw an exception
                time.sleep(delay) # delay for a while and try again
    return wrapper


@retry_on_error
def fusion(model, tokenizer, processor, image, question, choices, answer, index, output_dir): 
    responses = []

    sampling_params = SamplingParams(temperature=0.9, top_p=0.95, top_k=40, repetition_penalty=1.1, max_tokens=4096)

    gen_config = {
      'temperature': 0.9, 
      'top_p': 0.95, 
      'top_k': 40, 
      'repetition_penalty': 1.1
    }

    image_name = index
    file_dir = os.path.join(output_dir, "pic")
    os.makedirs(file_dir, exist_ok=True)
    image_path = os.path.join(file_dir, f"{image_name}.png")
    image.save(image_path)

    image_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text", 
                    "text": config_img_cn.system_prompt + "\nOriginal Question: \n" + question + "\nOriginal Answer: \n" + answer + "\n" + config_img_cn.user_prompt_subquestion
                },
            ],
        }
    ]
    image_inputs, video_inputs = process_vision_info(image_messages)
    text = processor.apply_chat_template(
        image_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    sub_questions = model.generate([text], sampling_params=sampling_params)[0].outputs[0].text

    image_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": config_img_cn.system_prompt + "\nSub Questions: \n" + sub_questions + "\n" + config_img_cn.user_prompt_multihop
                }
            ],
        }
    ]
    text = processor.apply_chat_template(
        image_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    multihop = model.generate([text], sampling_params=sampling_params)[0].outputs[0].text

    image_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text", 
                    "text": config_img_cn.system_prompt + "\nOriginal Question: \n" + question + "\nOriginal Answer: \n" + answer + "\nSub Questions: \n" + sub_questions + "\nMultihop: \n" + multihop + "\n" + config_img_cn.extract_prompt_qa
                }
            ],
        }
    ]
    text = processor.apply_chat_template(
        image_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    qa = model.generate([text], sampling_params=sampling_params)[0].outputs[0].text

    qa = extract_json(qa)
    qa["thinking"] = multihop

    qa["original_image"] = image_path
    
    responses.append(qa)

    return responses


def main(rank, gpu_id, per_num, shared_list, output_dir, filtered_data): 
    seed = 42 + gpu_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model_dir, torch_dtype="auto", device_map="auto"
    # )
    model = LLM(model=model_dir, seed=seed, tensor_parallel_size=8)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    
    multi_hop = []
    for i in tqdm(range(per_num), desc="Running Inference"):
        try:
            index = per_num * gpu_id + i + 1
            example = filtered_data[index]
            image = example['image']
            question = example['question']
            choices = ''.join(example['choices'])
            answer = example['solution']

            responses = fusion(model, tokenizer, processor, image, question, choices, answer, index, output_dir)
            multi_hop.extend(responses)

            with open(os.path.join(output_dir, f"thinking-cn-{per_num}_{rank}.json"), "w", encoding="utf-8") as f:
                json.dump(multi_hop, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print("Error: ", e)

    print("Total data: ", len(multi_hop))
    shared_list.extend(multi_hop)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description="Generate thinking data.")
    parser.add_argument('--example_num', type=int, required=True, help='Please enter the amount of training data')
    parser.add_argument('--num_gpus', type=int, required=True, help='Please enter the number of GPUs')
    parser.add_argument('--output_dir', type=str, required=True, help='Please enter the output directory')
    args = parser.parse_args()

    # ds = load_dataset("lmms-lab/multimodal-open-r1-8k-verified")
    ds = load_dataset("derek-thomas/ScienceQA")

    filtered_data = [item for item in ds['train'] if item.get("image") and item.get("question") and item.get("choices") and item.get("answer")]

    example_num = args.example_num
    num_gpus = args.num_gpus
    per_num = int(example_num / num_gpus)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    manager = multiprocessing.Manager()
    shared_list = manager.list()

    processes = []
    for rank in range(0, num_gpus):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        p = multiprocessing.Process(target=main, args=(rank, rank, per_num, shared_list, output_dir, filtered_data))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
      
    combined_results = list(shared_list)
    # print(combined_results)
    
    with open(os.path.join(output_dir, f"thinking-cn-{example_num}.json"), "w", encoding="utf-8") as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=4)
