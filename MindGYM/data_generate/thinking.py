from modelscope import snapshot_download
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import numpy as np
import base64
import argparse
import random
import string
import time
import torch
import config
from tqdm import tqdm
import multiprocessing
import os
from torch import cuda
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


model_dir = snapshot_download('Qwen/Qwen2.5-VL-72B-Instruct')


CATEGORIES = [
    "Mathematical Reasoning",
    "Scientific Knowledge",
    "Logical Deduction",
    "Technical Procedures",
    "Historical Events",
    "Ethical Considerations",
    "Economic Principles",
    "Psychological Insights"
]


class DataDeduplicator:
    def __init__(self, similarity_threshold=0.8):
        """
        Initialize the deduplication filter
        :param similarity_threshold: similarity threshold (0-1), if exceeded, it will be considered as a duplicate
        """
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.existing_docs = []  # 存储已有文档
        self.threshold = similarity_threshold
        
    def calculate_similarity(self, new_doc):
        """Calculate the maximum similarity between the new document and all existing documents"""
        if not self.existing_docs:
            return 0.0
            
        # Merge text to create TF-IDF matrix
        all_docs = self.existing_docs + [new_doc]
        tfidf_matrix = self.vectorizer.fit_transform(all_docs)
        
        # Calculate the similarity between the new document and all existing documents
        new_vec = tfidf_matrix[-1]
        existing_matrix = tfidf_matrix[:-1]
        
        if existing_matrix.shape[0] == 0:
            return 0.0
            
        similarities = cosine_similarity(new_vec, existing_matrix)
        return np.max(similarities)

    def is_duplicate(self, new_doc):
        """Determine whether it is repeated"""
        return self.calculate_similarity(new_doc) > self.threshold


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
def fusion(model, tokenizer, de_duplicator, category):
    responses = []

    sampling_params = SamplingParams(temperature=0.9, top_p=0.95, top_k=40, repetition_penalty=1.1, max_tokens=2048)

    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": config.user_prompt_background.format(category=category)}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    background = model.generate([text], sampling_params=sampling_params)

    if not de_duplicator.is_duplicate(background[0].outputs[0].text):
        de_duplicator.existing_docs.append(background[0].outputs[0].text)
    else: 
        raise Exception("Repeat the generation and try again")

    messages.extend([
        {"role": "system", "content": background[0].outputs[0].text}, 
        {"role": "user", "content": config.user_prompt_subquestion}
    ])
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    sub_questions = model.generate([text], sampling_params=sampling_params)

    messages.extend([
        {"role": "system", "content": sub_questions[0].outputs[0].text},
        {"role": "user", "content": config.user_prompt_multihop}
    ])
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    multihop = model.generate([text], sampling_params=sampling_params)

    messages.extend([
        {"role": "system", "content": multihop[0].outputs[0].text},
        {"role": "user", "content": config.extract_prompt_qa}
    ])
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    qa = model.generate([text], sampling_params=sampling_params)

    # QA = {**background, **sub_questions, **multihop}
    # QA['thinking'] = thinking
    qa = extract_json(qa[0].outputs[0].text)
    qa["thinking"] = multihop[0].outputs[0].text
    responses.append(qa)

    return responses


def main(output_dir, per_category_num): 
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = LLM(model=model_dir, seed=seed, tensor_parallel_size=8)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    dedup = DataDeduplicator(similarity_threshold=0.75)
    
    multi_hop = []
    for category in CATEGORIES:
        print(f"Generating documentation for {category}...")
        
        for i in tqdm(range(1, per_category_num + 1), desc="Running Inference"):
            try:
                responses = fusion(model, tokenizer, dedup, category)
                multi_hop.extend(responses)

                with open(os.path.join(output_dir, f"thinking_3.json"), "w", encoding="utf-8") as f:
                    json.dump(multi_hop, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print("Error: ", e)

    print("Total data: ", len(multi_hop))
    return multi_hop


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description="Generate thinking data.")
    parser.add_argument('--per_category_num', type=int, required=True, help='Please enter the amount of training data for each category')
    parser.add_argument('--output_dir', type=str, required=True, help='Please enter the output directory')
    args = parser.parse_args()

    per_category_num = args.per_category_num

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results = main(output_dir, per_category_num)
