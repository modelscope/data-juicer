import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import torch.nn.functional as F
import random
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate instruction pairs with specified parameters.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--selected_domain', type=str, required=True, help='Selected domain for generation')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save generated data')
    parser.add_argument('--gen_num', type=int, default=3, help='Number of pairs to generate')
    parser.add_argument('--warmup_file', type=str, required=True, help='Path to the warmup file')
    parser.add_argument('--similarity_threshold', type=float, default=0.85, help='Similarity threshold for filtering similar pairs')
    return parser.parse_args()

# 加载模型和分词器
args = parse_arguments()
model_name_or_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_embedding(text, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
    return embeddings

def generate_instruction_pair(prompt):
    messages = [
        {"role": "system", "content": "You are an AI assistant with a diverse and rich mindset."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    gen_kwargs = {
        "max_new_tokens": 2048,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 0.95,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id
    }
    
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, **gen_kwargs)
        generated_ids = generated_ids[:, len(model_inputs.input_ids[0]):]
    
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return response

def extract_instruction_parts(text):
    pattern = r"Instruction:\s*(.*?)(?=Input:|Output:|$)" \
              r"Input:\s*(.*?)(?=Output:|$)" \
              r"Output:\s*(.*)"
    
    match = re.search(pattern, text, re.DOTALL)
    if match:
        instruction, input_part, output = match.groups()
        return {
            'instruction': instruction.strip(),
            'input': input_part.strip(),
            'output': output.strip()
        }
    else:
        print("Failed to parse the generated text into parts.")
        return None

def load_warmup_data(file_path):
    instructions, inputs, outputs = [], [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                instructions.append(data.get('instruction', ''))
                inputs.append(data.get('input', ''))
                outputs.append(data.get('output', ''))
            except json.JSONDecodeError as e:
                print(f"Failed to parse line: {e}")
                print(f"Line content: {line}\n")
                continue
    return list(zip(instructions, inputs, outputs))

def generate_instruction_pairs(selected_domain, domains, warmup_file_path, num_pairs=3, similarity_threshold=0.85, save_dir=None):
    unselect_domains = [domain for domain in domains.keys() if domain != selected_domain]
    unselect_domains_str = ', '.join(unselect_domains)
    
    # 初始化warmup数据
    warmup_data = load_warmup_data(warmup_file_path)
    instructions, inputs, outputs = zip(*warmup_data) if warmup_data else ([], [], [])
    concatenated_strings = [f"Instruction: {i}, Input: {ip}, Output: {o}" for i, ip, o in zip(instructions, inputs, outputs)]
    embeddings_list = [get_embedding(cs) for cs in concatenated_strings]

    prompt_template = (
        "You are an AI model with expertise in {}. Here's a brief description of this domain:\n"
        "{}\n\n"
        "Generate only an instruction pair related to this field.\n\n"
        "The response should include three parts:\n\n"
        "Instruction: A clear command or question that can be understood by the assistant.\n"
        "Input: Any information provided to help it understand the instruction. If there is no need to generate, just keep empty.\n"
        "Output: The expected answer or action.\n\n"
        "Keep the generated content focused on {}. And do not involve {} related knowledge.\n\n"
        "Note that you should generate content strongly unrelated and different to these examples to ensure diversity in the generated output:\n"
        "Counterexample: {}\n\n"
        "The format of the generated content should be: Instruction: [], Input: [], Output: []."
    )
    
    instruction_pairs = []
    generated_count = 0
    attempts = 0
    
    while generated_count < num_pairs:
        attempts += 1
        
        sample_examples = random.sample(concatenated_strings, min(3, len(concatenated_strings)))
        examples_str = '\n'.join(sample_examples)
        
        
        print(f"Reference Examples for Pair {generated_count + 1}:\n{examples_str}\n")

        prompt = prompt_template.format(
            selected_domain,
            domains[selected_domain],
            selected_domain,
            unselect_domains_str,
            examples_str
        )
        raw_response = generate_instruction_pair(prompt)
        
        print(f"Raw Response {attempts}: {raw_response}\n")
        
        extracted_pair = extract_instruction_parts(raw_response)
        if extracted_pair:
            new_concatenated_string = f"Instruction: {extracted_pair['instruction']}, Input: {extracted_pair['input']}, Output: {extracted_pair['output']}"
            new_embedding = get_embedding(new_concatenated_string)

            similarities = [F.cosine_similarity(torch.tensor(new_embedding), torch.tensor(old_emb), dim=-1).item() for old_emb in embeddings_list]
            print(f"Similarities to existing pairs: {similarities}")

            if all(sim < similarity_threshold for sim in similarities):
                instruction_pairs.append(extracted_pair)
                generated_count += 1
                
                instructions = list(instructions) + [extracted_pair['instruction']]
                inputs = list(inputs) + [extracted_pair['input']]
                outputs = list(outputs) + [extracted_pair['output']]
                concatenated_strings.append(new_concatenated_string)
                embeddings_list.append(new_embedding)

    if save_dir:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        with open(save_dir, 'w', encoding='utf-8') as f:
            for pair in instruction_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        print(f"Saved {len(instruction_pairs)} pairs to {save_dir}")
    
    return instruction_pairs

# Domain description
domains = {
    "common_sense": "Common sense generally includes a knowledge-based question and its corresponding answer, without reasoning.",
    "reasoning": "Reasoning involves the ability to think logically about a situation or problem, to draw conclusions from available information, and to apply knowledge in new situations.",
    "mathematics": "Mathematical skills include the ability to perform calculations, understand mathematical concepts, solve hard and professional math problems, and apply mathematical reasoning.",
    "coding": "Design and generate specific code programs, or apply algorithms and data structures, with code generation in the Output."
}

instruction_pairs = generate_instruction_pairs(
    selected_domain=args.selected_domain,
    domains=domains,
    warmup_file_path=args.warmup_file,
    num_pairs=args.gen_num,
    similarity_threshold=args.similarity_threshold,
    save_dir=args.save_dir
)

for idx, pair in enumerate(instruction_pairs, start=1):
    print(f"Generated Pair {idx}:")
    print(f"Instruction: {pair['instruction']}")
    print(f"Input: {pair['input']}")
    print(f"Output: {pair['output']}\n")