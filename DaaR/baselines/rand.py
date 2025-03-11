import json
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Select and shuffle 8000 data points, save to new files.")
parser.add_argument('--input_file_path', type=str, default='./data/raw/40k_data.jsonl', help="Path to the input 40k_data.jsonl file.")
parser.add_argument('--input_tsne_path', type=str, default='./data/res/qw25/baselines/rand', required=True, help="Path to the input tsne_data.npz file.")
parser.add_argument('--output_path', type=str, required=True, help="Output directory to save the selected data and plot.")
args = parser.parse_args()

# Define output file paths
os.makedirs(args.output_path, exist_ok=True)
output_file_path = os.path.join(args.output_path, '8k_selected_data.jsonl')
output_npz_file_path = os.path.join(args.output_path, '8k_tsne_data.npz')

# Random seed
random.seed(42)

# Read JSONL data and load into a list
with open(args.input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Parse each line as a JSON object and add to a list
data = [json.loads(line.strip()) for line in lines]

# Load the NPZ file
npz_data = np.load(args.input_tsne_path)
embedding_2d = npz_data['embeddings_2d']
labels = npz_data['labels']

# Ensure the number of data points matches between JSONL and NPZ
assert len(data) == embedding_2d.shape[0] == labels.shape[0], "Mismatch in data lengths"

# Randomly select 8000 indices
selected_indices = random.sample(range(len(data)), 8000)

# Select data and embeddings based on the random indices
selected_data = [data[i] for i in selected_indices]
selected_embedding_2d = embedding_2d[selected_indices]
selected_labels = labels[selected_indices]

# Shuffle the selected data and embeddings together
combined = list(zip(selected_data, selected_embedding_2d, selected_labels))
random.shuffle(combined)
selected_data, selected_embedding_2d, selected_labels = zip(*combined)

# Write the selected JSON data to a new file
with open(output_file_path, 'w', encoding='utf-8') as file:
    for item in selected_data:
        file.write(json.dumps(item, ensure_ascii=False) + '\n')

# Save the selected numpy arrays to an NPZ file
np.savez(output_npz_file_path, embeddings_2d=np.array(selected_embedding_2d), labels=np.array(selected_labels))

print(f"Successfully selected and shuffled 8000 data points from {args.input_file_path} and {args.input_tsne_path}, saved to {output_file_path} and {output_npz_file_path}")

# t-SNE draw
file_root = args.output_path
file_path = os.path.join(file_root, '8k_tsne_data.npz')
data = np.load(file_path)

embeddings_2d = data['embeddings_2d']
labels = data['labels']

label_to_color = {
    "code-en": "red",
    "cot-en": "green",
    "dolly-en": "blue",
    "math-en": "purple"
}

plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    indices = np.where(labels == label)
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], c=label_to_color[label], label=label, s=3, alpha=0.7)

plt.legend()
plt.title('t-SNE Visualization of Selected Data Samples')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

path_save = os.path.join(file_root, '8k_tsne_data.png')
plt.savefig(path_save)
plt.close()
