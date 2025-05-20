import os
import json
import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

def create_directories(output_dir):
    txt_output_dir = os.path.join(output_dir, "txt")
    jsonl_output_dir = os.path.join(output_dir, "jsonl")
    os.makedirs(txt_output_dir, exist_ok=True)
    os.makedirs(jsonl_output_dir, exist_ok=True)
    return txt_output_dir, jsonl_output_dir

domains = {
    "common_sense": "Common sense generally includes a knowledge-based question and its corresponding answer, without reasoning.",
    "reasoning": "Reasoning involves the ability to think logically about a situation or problem, to draw conclusions from available information, and to apply knowledge in new situations.",
    "mathematics": "Mathematical skills include the ability to perform calculations, understand mathematical concepts, solve hard and professional math problems, and apply mathematical reasoning.",
    "coding": "Design and generate specific code programs, or apply algorithms and data structures, with code generation in the Output."
}

def create_prompt(selected_domain):
    unselect_domains = [domain for domain in domains.keys() if domain != selected_domain]
    unselect_domains_str = ", ".join(unselect_domains[:-1]) + (f", and {unselect_domains[-1]}" if len(unselect_domains) > 1 else unselect_domains[0])
    return f"""
You are an AI model with expertise in {selected_domain}. Here's a brief description of this domain:
{domains[selected_domain]}

Generate 5 different instruction pairs related to this field with various lengths. Maintain the index format: Instruction [1 ... 5].

The response should include three parts:

1. Instruction: A clear command or question that can be understood by the assistant.
2. Input: Any information provided to help it understand the instruction. If there is no need to generate, just keep empty.
3. Output: The expected answer or action.

Ensure the format is strictly as follows for each pair:

Instruction [X]: [Your instruction here]
Input: [Your input here or leave empty]
Output: [Your output here]

Keep the generated content focused on {selected_domain}. And do not involve {unselect_domains_str} related knowledge.
"""

def generate_instruction_pairs(prompt, tokenizer, model, num_pairs=1, max_new_tokens=2048):
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 0.9,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id
    }
    
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, **gen_kwargs)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses

def warm_up_stage(selected_domain, tokenizer, model, txt_output_dir, jsonl_output_dir):
    raw_output_file = os.path.join(txt_output_dir, f"{selected_domain}_warmup.txt")
    jsonl_output_file = os.path.join(jsonl_output_dir, f"{selected_domain}_warmup.jsonl")
    
    prompt = create_prompt(selected_domain)
    warm_up_data = []

    for i in tqdm(range(1), desc="Generating instruction pairs"):
        pairs = generate_instruction_pairs(prompt, tokenizer, model, num_pairs=1)
        if pairs:
            warm_up_data.extend(pairs)

    with open(raw_output_file, 'w') as f:
        for entry in warm_up_data:
            clean_entry = entry.strip()
            f.write(clean_entry + '\n\n')

    # May be failed
    extracted_pairs = []
    for entry in warm_up_data:
        clean_entry = entry.strip()
        blocks = re.split(r'\n\n+', clean_entry)
        for block in blocks:
            match = re.match(
                r'Instruction $$(\d+)$$:\s*(.*?)\nInput:\s*(.*?)\nOutput:\s*(.*)',
                block.strip(),
                re.DOTALL
            )
            if match:
                _, instr, inp, outp = match.groups()
                pair = {
                    "instruction": instr.strip(),
                    "input": inp.strip() if inp.strip() else "",
                    "output": outp.strip()
                }
                extracted_pairs.append(pair)

    with open(jsonl_output_file, 'w') as f:
        for pair in extracted_pairs:
            f.write(json.dumps(pair) + '\n')
    print(f"Data saved to {jsonl_output_file}")

def main():
    parser = argparse.ArgumentParser(description="Warm up the model and generate instruction pairs.")
    parser.add_argument('--output_dir', type=str, default='./daar/1_centroid/warmup_seed/qw25')
    parser.add_argument('--model_path', type=str, default='./models/Qwen2.5-7B')
    args = parser.parse_args()

    txt_output_dir, jsonl_output_dir = create_directories(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", trust_remote_code=True)

    for domain in domains.keys():
        warm_up_stage(domain, tokenizer, model, txt_output_dir, jsonl_output_dir)

if __name__ == "__main__":
    main()