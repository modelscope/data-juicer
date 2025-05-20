import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description="Generate and plot embeddings.")
parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model.")
parser.add_argument('--output_path', type=str, required=True, help="Path to save the output files.")
args = parser.parse_args()

# Random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Loading pre-trained models and tokenizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Max length
max_length = 512

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        # Extraction from Embedding Layer
        embeddings = model.embed_tokens(inputs['input_ids']).mean(dim=1).cpu().numpy()  # Average tokens
    return embeddings

def plot_and_save_embeddings(embeddings, labels, output_folder):
    # Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, random_state=seed, verbose=2)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Save t-SNE
    tsne_save_path = os.path.join(output_folder, 'tsne_data.npz')
    np.savez_compressed(tsne_save_path, embeddings_2d=embeddings_2d, labels=labels)
    
    # Save raw embeddings
    embeddings_save_path = os.path.join(output_folder, 'embeddings_data.npz')
    np.savez_compressed(embeddings_save_path, embeddings=embeddings, labels=labels)

    # Scatter plot
    label_to_color = {
        "code-en": "red",
        "cot-en": "green",
        "dolly-en": "blue",
        "math-en": "purple"
    }

    plt.figure(figsize=(10, 8))

    for label, color in label_to_color.items():
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], c=color, label=label, s=3)
    
    plt.legend()
    plot_save_path = os.path.join(output_folder, 'tsne.png')
    plt.savefig(plot_save_path)
    plt.close()

output_folder = args.output_path
os.makedirs(output_folder, exist_ok=True)

data_folder = './data/raw'
filenames = ['code-en.jsonl', 'cot-en.jsonl', 'dolly-en.jsonl', 'math-en.jsonl']
merged_file = os.path.join(data_folder, '40k_data.jsonl')

embeddings = []
labels = []

# Process each file separately
temp_data = []
for filename in filenames:
    file_path = os.path.join(data_folder, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Processing {filename}"):
            data = json.loads(line)
            text = f"{data['instruction']} {data['input']} {data['output']}"
            embedding = get_embedding(text)
            # Append embedding and filename (label)
            embeddings.append(embedding)
            labels.append(filename.split('.')[0])
            # Append to temp data for merging
            temp_data.append({
                'instruction': data['instruction'],
                'input': data['input'],
                'output': data['output']
            })

# Concatenate embeddings
embeddings = np.concatenate(embeddings, axis=0)

# Plot and save embeddings
plot_and_save_embeddings(embeddings, labels, output_folder)

# After processing, merge data and store in final file
with open(merged_file, 'w', encoding='utf-8') as outfile:
    for entry in temp_data:
        json.dump(entry, outfile)
        outfile.write('\n')

print('Processed data and embeddings successfully saved.')
