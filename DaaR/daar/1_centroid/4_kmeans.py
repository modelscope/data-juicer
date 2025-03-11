import os
import json
import numpy as np
from sklearn.cluster import KMeans
import argparse

def load_npz_labels(file_path):
    """Load 'labels' data from a .npz file."""
    with np.load(file_path) as data:
        return data['labels']

def add_labels_to_jsonl(data_file, labels, output_file):
    """Add labels to each record in the JSONL file and save to a new file."""
    with open(data_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            record = json.loads(line.strip())
            record['label'] = int(labels[i])  # Ensure the label is of integer type
            outfile.write(json.dumps(record) + '\n')

def cluster_embeddings(args):
    """Cluster embeddings using KMeans with predefined centers and save labels."""
    # Load the .npz file
    data = np.load(args.input_ebd)

    # Extract embeddings data
    if 'embeddings' in data:
        embeddings = data['embeddings']
    else:
        raise ValueError("Array 'embeddings' not found in the file.")

    # Obtain the centers for the four domains
    domain_centers = []
    domains = ['code', 'cot', 'dolly', 'math']

    for domain in domains:
        domain_file_path = os.path.join(args.input_seed, f'{domain}_ebd.npz')
        domain_data = np.load(domain_file_path)
        
        if 'mean_embedding' in domain_data:
            domain_center = domain_data['mean_embedding']
            domain_centers.append(domain_center)
        else:
            raise ValueError(f"Array 'mean_embedding' not found in {domain_file_path}.")

    initial_centers = np.vstack(domain_centers)

    # Perform clustering using predefined initial centers
    n_clusters = len(domains)  # Number of clusters matches the number of provided centers
    kmeans_manual_init = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1, random_state=42)
    kmeans_manual_init.fit(embeddings)

    # Save manually initialized labels
    os.makedirs(args.output_path, exist_ok=True)
    manual_labels_file_path = os.path.join(args.output_path, 'k_labels.npz')
    np.savez(manual_labels_file_path, labels=kmeans_manual_init.labels_)
    print(f"Manual initialization labels saved to {manual_labels_file_path}")

    return kmeans_manual_init.labels_

def main():
    parser = argparse.ArgumentParser(description="Cluster embeddings and add labels to JSONL data.")
    parser.add_argument('--input_ebd', type=str, default='./daar/1_centroid/train_data/qw25/ebd.npz', help='Path to the embeddings npz file.')
    parser.add_argument('--input_seed', type=str, default='./daar/1_centroid/syn/qw25/ebd', help='Path to the directory containing domain center npz files.')
    parser.add_argument('--data_path', type=str, default='./daar/1_centroid/train_data/qw25/raw_data.jsonl', help='Path to the input JSONL data file.')
    parser.add_argument('--output_path', type=str, default='./daar/1_centroid/train_data/qw25', help='Output directory path for saving clustered labels and labeled JSONL file.')
    args = parser.parse_args()

    # Perform clustering and get labels
    labels = cluster_embeddings(args)

    # Add labels to JSONL file
    output_jsonl_file = os.path.join(args.output_path, 'train_data.jsonl')
    add_labels_to_jsonl(args.data_path, labels, output_jsonl_file)
    print(f"Labels added and saved to {output_jsonl_file}")

if __name__ == '__main__':
    main()
