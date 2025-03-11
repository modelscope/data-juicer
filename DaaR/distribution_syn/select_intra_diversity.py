import os
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import jsonlines

def main(args):
    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Construct full paths for embedding and t-SNE data files
    embedding_file_path = os.path.join(args.input_path, 'embeddings_data.npz')
    tsne_file_path = os.path.join(args.input_path, 'tsne_data.npz')

    # Load embeddings and t-SNE data
    embedding_data = np.load(embedding_file_path)
    embeddings = embedding_data['embeddings']
    labels = embedding_data['labels']

    tsne_data = np.load(tsne_file_path)
    tsne_embeddings = tsne_data['embeddings_2d']

    # Convert to PyTorch tensors
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Define category index ranges
    category_ranges = {
        'code-en': (0, 10000),
        'cot-en': (10000, 20000),
        'dolly-en': (20000, 30000),
        'math-en': (30000, 40000),
    }

    # Compute semantic centers for each category
    centers = {label: embeddings_tensor[start:end].mean(dim=0)
               for label, (start, end) in category_ranges.items()}

    # Select highest similarity data points for each category
    selected_indices = {}
    for label, (start, end) in category_ranges.items():
        current_embeddings = embeddings_tensor[start:end]
        center = centers[label]
        similarities = F.cosine_similarity(current_embeddings, center.unsqueeze(0)).squeeze().tolist()
        selected_indices[label] = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[args.lower:args.upper]

    # Collect selected data
    selected_embeddings = []
    selected_labels = []
    selected_tsne_embeddings = []

    for label, (start, end) in category_ranges.items():
        indices = selected_indices[label]
        selected_embeddings.extend(embeddings[start:end][indices])
        selected_tsne_embeddings.extend(tsne_embeddings[start:end][indices])
        selected_labels.extend([label] * len(indices))

    # Save selected embeddings and t-SNE data
    np.savez_compressed(os.path.join(args.output_path, '8k_embeddings_data.npz'),
                        embeddings=selected_embeddings, labels=selected_labels)
    np.savez_compressed(os.path.join(args.output_path, '8k_tsne_data.npz'),
                        embeddings_2d=selected_tsne_embeddings, labels=selected_labels)

    # Extract corresponding instruction pairs from original JSONL files
    for label, (start, end) in category_ranges.items():
        file_path = os.path.join(args.data_folder, f"{label}.jsonl")
        output_file_path = os.path.join(args.output_path, f"{label}.jsonl")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            with open(output_file_path, 'w', encoding='utf-8') as out_f:
                for local_index in selected_indices[label]:
                    out_f.write(lines[local_index])

    print("Data successfully processed and saved.")

    # Merge files into a single JSONL
    output_file = os.path.join(args.output_path, '8k_selected_data.jsonl')
    file_names = ['code-en.jsonl', 'cot-en.jsonl', 'dolly-en.jsonl', 'math-en.jsonl']

    all_records = []
    for file_name in file_names:
        with jsonlines.open(os.path.join(args.output_path, file_name)) as reader:
            all_records.extend(reader)

    random.shuffle(all_records)

    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(all_records)

    print(f"Merge complete. Total records: {len(all_records)}")

    # Remove original files after merging
    for file_name in file_names:
        os.remove(os.path.join(args.output_path, file_name))

    # Plot t-SNE
    plot_tsne(args.output_path)

def plot_tsne(output_path):
    file_path = os.path.join(output_path, '8k_tsne_data.npz')
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
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                    c=label_to_color[label], label=label, s=3, alpha=0.7)

    plt.legend()
    plt.title('t-SNE Visualization of Selected Data Samples')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    path_save = os.path.join(output_path, '8k_tsne_data.png')
    plt.savefig(path_save)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and visualize data.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the folder containing embedding and t-SNE data files.")
    parser.add_argument('--data_folder', type=str, default='./data/raw', help="Path to the folder containing raw data.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the folder for saving processed data.")
    parser.add_argument('--lower', type=int, required=True, help="Lower bound for selecting indices.")
    parser.add_argument('--upper', type=int, required=True, help="Upper bound for selecting indices.")
    args = parser.parse_args()

    main(args)
