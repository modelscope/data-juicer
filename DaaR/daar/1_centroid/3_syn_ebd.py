import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

# Set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define maximum length
max_length = 512

# Define mapping for output filenames
output_filename_mapping = {
    'coding.jsonl': 'code_ebd.npz',
    'common_sense.jsonl': 'dolly_ebd.npz',
    'mathematics.jsonl': 'math_ebd.npz',
    'reasoning.jsonl': 'cot_ebd.npz'
}

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        # Directly use the embed_tokens layer of the model for conversion
        embeddings = model.embed_tokens(inputs['input_ids']).mean(dim=1).cpu().numpy()  # Take the mean to simplify representation
    return embeddings

def process_file(input_file_path, output_folder, tokenizer, model):
    print(f"Processing file: {input_file_path}")
    embeddings_list = []

    with open(input_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Extracting embeddings")):
            try:
                data = json.loads(line.strip())
                concatenated_text = f"{data.get('instruction', '')} {data.get('input', '')} {data.get('output', '')}"
                embedding = get_embedding(concatenated_text, tokenizer, model)
                embeddings_list.append(embedding)
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON on line {i+1}: {e}")
                continue  # Skip problematic lines

    if not embeddings_list:
        print(f"No valid entries found in {input_file_path}. Skipping.")
        return

    embeddings_array = np.vstack(embeddings_list)
    mean_embedding = embeddings_array.mean(axis=0)

    base_filename = os.path.basename(input_file_path)
    output_file = os.path.join(output_folder, output_filename_mapping.get(base_filename, f"{os.path.splitext(base_filename)[0]}_ebd.npz"))
    np.savez_compressed(output_file, mean_embedding=mean_embedding)

    print(f"Mean embedding saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process JSONL files and extract mean embeddings.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory.')
    args = parser.parse_args()

    input_folder = './daar/1_centroid/syn/qw25'
    output_folder = './daar/1_centroid/syn/qw25/ebd'

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(device)

    # Check and set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Iterate over all .jsonl files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jsonl'):
            input_file_path = os.path.join(input_folder, filename)
            process_file(input_file_path, output_folder, tokenizer, model)

if __name__ == '__main__':
    main()
